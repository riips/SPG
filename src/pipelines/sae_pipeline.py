"""sae_pipeline.py
Pipeline that applies an SAE on top of encoder tokens and provides
anomaly scores and maps, along with optional analysis metrics.
"""

import logging
import math
from typing import List, Tuple, Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import ModelCfg, WeightModuleCfg, SAECfg
from data import ImageRecord
from encoders.base_encoder import BaseEncoder
from modules import WeightModule

from modules.sae import create_sae, BatchOutput

from .base_pipeline import BasePipeline
from .core.contracts import (
    Extractor,
    AnomalyScorer,
    LossComputer,
    ScoreOptions,
    AnomalyResult,
    AnomalyContext,
)
from .core.tensor_ops import (
    softmax_with_temperature,
    get_score_map_with_target_size,
    loss_function,
)

log = logging.getLogger(__name__)


class _MSEAccumulator:
    """Accumulates mean squared error over all tokens and dimensions."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, outputs: BatchOutput) -> None:
        diff = (outputs.x.detach() - outputs.x_hat.detach()).pow(2)
        self.sum += float(diff.sum().item())
        self.count += diff.numel()

    def finalize(self) -> float:
        return self.sum / self.count if self.count > 0 else float("nan")


class _R2Accumulator:
    """Accumulates SSE/SST for R²."""

    def __init__(self) -> None:
        self.sse = 0.0
        self.sst = 0.0

    def update(self, outputs: BatchOutput) -> None:
        x = outputs.x.detach()
        x_hat = outputs.x_hat.detach()
        self.sse += float((x - x_hat).pow(2).sum().item())
        mean = x.mean()
        self.sst += float((x - mean).pow(2).sum().item())

    def finalize(self) -> float:
        if self.sst <= 0:
            return float("nan")
        return 1.0 - (self.sse / (self.sst + 1e-8))


class _ZL1Accumulator:
    """Accumulates mean L1 norm of latent z."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, outputs: BatchOutput) -> None:
        z = outputs.z.detach()
        self.sum += float(z.abs().sum().item())
        self.count += z.numel()

    def finalize(self) -> float:
        return self.sum / self.count if self.count > 0 else float("nan")


class _ZSparsityAccumulator:
    """Accumulates fraction of near-zero activations in z."""

    def __init__(self, threshold: float = 1e-3) -> None:
        self.threshold = threshold
        self.sum = 0.0
        self.count = 0

    def update(self, outputs: BatchOutput) -> None:
        z = outputs.z.detach()
        mask = (z.abs() < self.threshold).float()
        self.sum += float(mask.sum().item())
        self.count += mask.numel()

    def finalize(self) -> float:
        return self.sum / self.count if self.count > 0 else float("nan")


class _ZActivationAccumulator:
    """Tracks per-dimension activation hits to detect dead atoms (|z| > threshold)."""

    def __init__(self, threshold: float = 1e-4) -> None:
        self.threshold = threshold
        self.active_hits: torch.Tensor | None = None  # counts of batches where dim was active
        self.num_updates: int = 0

    def update(self, outputs: BatchOutput) -> None:
        z = outputs.z.detach()
        # z shape: (B, T, D)
        if z.ndim != 3:
            return
        B, T, D = z.shape
        if self.active_hits is None:
            self.active_hits = torch.zeros(D, dtype=torch.long)
        batch_active = (z.abs() > self.threshold).any(dim=(0, 1)).long()  # (D,)
        self.active_hits += batch_active.cpu()
        self.num_updates += 1

    def finalize(self) -> Dict[str, object]:
        if self.active_hits is None:
            return {
                "active_per_dim": [],
                "dead_per_dim": [],
                "active_global": 0,
                "dead_global": 0,
                "num_updates": self.num_updates,
            }
        active_per_dim = self.active_hits.tolist()
        dead_per_dim = [1 if h == 0 else 0 for h in active_per_dim]
        dead_global = int(sum(dead_per_dim))
        active_global = int(len(active_per_dim) - dead_global)
        return {
            # "active_per_dim": active_per_dim,
            # "dead_per_dim": dead_per_dim,
            "active_global": active_global,
            "dead_global": dead_global,
            "num_updates": self.num_updates,
        }


class SAEPipeline(BasePipeline):
    """Pipeline that leverages two SAEs for anomaly detection."""

    def __init__(
        self, model_cfg: ModelCfg, encoder: BaseEncoder, device: torch.device | str
    ) -> None:
        """Initializes sub-modules and caches frequently used config.

        Args:
            model_cfg: Hydra / OmegaConf configuration for the model.
            encoder: Frozen backbone network with `encode_image`.
            device: Target device (``"cpu"`` or ``"cuda:0"`` etc.).
        """
        super().__init__(model_cfg, encoder, device)

        sae_cfg: List[SAECfg] = model_cfg.method_config.sae  # type: ignore
        # self._check_sae_cfg(sae_cfg)
        
        self.sae_cfg = sae_cfg
        self.sae_list = [create_sae(sae_cfg[i], encoder.output_dim) for i in range(len(sae_cfg))]

        self.hidden_dim_anomaly = getattr(sae_cfg[0], "hidden_dim_anomaly", None)
        self.separate_loss_weight = getattr(model_cfg.method_config, "separate_loss_weight", 0.0)
        self.separate_loss_mask = getattr(model_cfg.method_config, "separate_loss_mask", "normal_ratio")
        self.separate_loss_zero_eps = getattr(model_cfg.method_config, "separate_loss_zero_eps", 1e-6)

        self.anomaly_map_shift = getattr(model_cfg.method_config, "anomaly_map_shift", 0.0)
        self.anomaly_map_scale = getattr(model_cfg.method_config, "anomaly_map_scale", 1.0)

        if model_cfg.optional_modules and model_cfg.optional_modules.weight_module:
            weight_cfg: WeightModuleCfg = model_cfg.optional_modules.weight_module
            self.weight_module = WeightModule(weight_cfg, encoder)
        else:
            self.weight_module = None

        self._modules = nn.ModuleDict(
            {
                "sae_object": self.sae_list[0],
            }
        )
        if self.weight_module is not None:
            self._modules["weight_module"] = self.weight_module

        # Metric registry (extensible) and state buffers
        self._metric_builders: Dict[str, callable] = {}
        self._metrics: Dict[str, Dict[str, object]] = {}
        self._metrics_enabled: bool = False
        self._register_builtin_metrics()
        self.reset_metrics()
    
    def _check_sae_cfg(self, sae_cfg: List[SAECfg]) -> None:
        assert len(sae_cfg) == 1, "One SAE is required"
        assert sae_cfg[0].hidden_dim_anomaly is not None, "hidden_dim_anomaly is required"
        return

    def build_components(
        self, model_cfg: ModelCfg, encoder: BaseEncoder
    ) -> Tuple[Extractor, AnomalyScorer, LossComputer]:
        pipeline = self

        class Extractor:
            def extract(
                self, inputs: ImageRecord
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                image_features, patch_features_list = pipeline._get_features(inputs)
                return image_features, patch_features_list

        class AnomalyScorer:
            def compute(
                self,
                image_features: torch.Tensor,
                patch_features_list: List[torch.Tensor],
                need_context: bool,
                score_options: ScoreOptions,
            ) -> AnomalyResult:

                weights = pipeline._compute_weight(image_features)

                _layer = -1
                x = torch.cat([image_features.view(image_features.shape[0], 1, -1), patch_features_list[_layer][:, 1:]], dim=1)

                assert x.ndim == 3
                # assert x.shape[0] == len(inputs.img) # batch size
                assert x.shape[1] == patch_features_list[_layer][:, 1:].shape[1] + 1, f"{x.shape[1]} != {patch_features_list[_layer][:, 1:].shape[1]} + 1" # number of patches + 1 (image feature)
                assert x.shape[2] == pipeline.sae_list[0].input_dim
                # ------------------------------------------------------------

                outputs: BatchOutput = pipeline.sae_list[0](x)
                outputs_dict: Dict[str, BatchOutput] = {"obj": outputs}

                # Update metrics automatically if enabled
                pipeline._update_metrics_if_enabled(outputs_dict)

                if need_context:
                    context = AnomalyContext(
                        image_features=image_features,
                        patch_features_list=patch_features_list,
                        weights=weights,
                        outputs=outputs_dict["obj"],
                    )
                else:
                    context = AnomalyContext()  # = {}
                
                anomaly_score = outputs.z[:, 0, :pipeline.hidden_dim_anomaly].norm(dim=-1).detach().cpu()
                anomaly_map = outputs.z[:, 1:, :pipeline.hidden_dim_anomaly].norm(dim=-1)

                anomaly_map = get_score_map_with_target_size(anomaly_map, pipeline.cfg.image_size)
                anomaly_map = torch.from_numpy(gaussian_filter(anomaly_map.detach().cpu(), sigma=pipeline.cfg.sigma))  # type: ignore

                return AnomalyResult(
                    context=context,
                    anomaly_score=anomaly_score,
                    anomaly_map=anomaly_map,
                )

        class LossComputer:
            def compute(
                self, result: AnomalyResult, label: torch.Tensor, gt: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
                assert "context" in result, "missing context in result"
                ctx: AnomalyContext = result["context"]

                assert "outputs" in ctx, "missing outputs in context"
                assert "weights" in ctx, "missing weights in context"

                return pipeline._compute_loss(
                    ctx["outputs"],
                    label,
                    gt,
                    ctx["weights"],
                )

        return Extractor(), AnomalyScorer(), LossComputer()

    def _compute_weight(
        self, image_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Returns per-layer weights.

        Args:
            image_features: Global image features (optional, used by
                some weight modules).

        Returns:
            torch.Tensor: Tensor of shape ``(B, num_layers)`` or
            ``(1, num_layers)`` if no module is present.
        """
        if self.weight_module is not None:
            return self.weight_module(image_features)
        else:
            weights = torch.ones(
                (1, len(self.sae_cfg)), device=self.device
            )
            return weights

    def _get_features(
        self, inputs: ImageRecord
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extracts image-level and patch-level features from the backbone.

        Args:
            inputs: Data container. If it already contains pre-computed
                ``feats`` and ``mid_feats`` they are used directly.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: ``(image_features, patch_features_list)``.
        """
        if isinstance(inputs.feats, torch.Tensor) and isinstance(
            inputs.mid_feats, list
        ):
            inputs.to(self.device)
            image_features = inputs.feats
            patch_features_list = inputs.mid_feats
        else:
            image = inputs.img
            with torch.no_grad():
                image_features, patch_features_list = self.encoder.encode_image(
                    image=image,
                    features_list=self.cfg.features_list,
                )

        return image_features, patch_features_list

    def _compute_loss(
        self,
        outputs: BatchOutput,
        label: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates detection/segmentation loss with SAE recon errors.

        - CLS token is weighted by image-level `label`.
        - Patch tokens are weighted by per-patch anomaly ratios from `gt`.

        Args:
            outputs_obj: Outputs of object SAE.
            outputs_ano: Outputs of anomaly SAE.
            label: Binary labels ``(B,)`` indicating anomaly.
            gt: Pixel-wise binary ground-truth ``(B, H, W)``.
            weights: Per-layer weights ``(B, L)`` (unused here, kept for API parity).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(det_loss, seg_loss)`` where
            seg_loss is averaged over tokens.
        """
        assert label is not None, "label is required"
        assert gt is not None, "gt is required"
        assert outputs is not None, "outputs is required"
        assert weights is not None, "weights is required"

        inputs = outputs.x
        assert inputs.shape == outputs.x_hat.shape, f"{inputs.shape} != {outputs.x_hat.shape}"

        num_patches_side = int(np.sqrt(outputs.x.shape[1] - 1))
        patch_gt = self._make_patch_gt(gt, num_patches_side)
        expected_len = num_patches_side * num_patches_side
        assert patch_gt.shape == (inputs.shape[0], expected_len), f"{patch_gt.shape} != {(inputs.shape[0], expected_len)}"

        z_obj = outputs.z[:, :, self.hidden_dim_anomaly:]
        dictionary_obj = self.sae_list[0].decoder.dictionary.matrix[self.hidden_dim_anomaly:]
        x_hat_obj = z_obj @ dictionary_obj

        z_ano = outputs.z[:, :, :self.hidden_dim_anomaly]
        dictionary_ano = self.sae_list[0].decoder.dictionary.matrix[:self.hidden_dim_anomaly]
        x_hat_ano = z_ano @ dictionary_ano
        
        rec_loss_normal = (inputs - x_hat_obj).pow(2).mean(dim=-1)
        # rec_loss_anom = (inputs - outputs_ano.x_hat).pow(2).mean(dim=-1)
        rec_loss_anom = (inputs - (x_hat_obj.detach() + x_hat_ano)).pow(2).mean(dim=-1)
        # rec_loss_anom = (inputs - (outputs.x_hat)).pow(2).mean(dim=-1)
        # rec_loss_anom = (inputs - (outputs_obj.x_hat.detach() + outputs_ano.x_hat)).pow(2).mean(dim=-1)

        # prepend image-level label (CLS token) then patch-wise weights to match token length
        cls_weight = label.view(-1, 1).float()
        weight = torch.cat([cls_weight, patch_gt], dim=1)
        assert weight.shape == rec_loss_normal.shape, f"{weight.shape} != {rec_loss_normal.shape}"

        recon_loss = (1 - weight) * rec_loss_normal + weight * rec_loss_anom

        if self.separate_loss_mask in ("normal_ratio"):
            sep_mask = 1 - weight
        elif self.separate_loss_mask in ("zero_only"):
            sep_mask = (weight < self.separate_loss_zero_eps).float()
        else:
            raise ValueError(f"Unknown separate_loss_mask: {self.separate_loss_mask}")

        separate_loss = sep_mask * outputs.z[:, :, -self.hidden_dim_anomaly:].norm(p=1, dim=-1)

        return {
            "recon_loss": recon_loss.mean(),
            "separate_loss": separate_loss.mean() * self.separate_loss_weight,
        }

    # ------------------------------ Metrics API ------------------------------ #
    def _register_builtin_metrics(self) -> None:
        """Register built-in SAE metrics."""

        def register(name: str, builder: callable) -> None:
            self._metric_builders[name] = builder

        register("mse", _MSEAccumulator)
        register("r2", _R2Accumulator)
        register("z_l1", _ZL1Accumulator)
        register("z_sparsity", _ZSparsityAccumulator)
        register("z_activation", _ZActivationAccumulator)

    def register_metric(self, name: str, builder: callable) -> None:
        """Allow external registration of custom metric accumulators."""
        self._metric_builders[name] = builder

    def _normalize_outputs_for_metrics(
        self,
        outputs: BatchOutput | Sequence[BatchOutput] | Mapping[str, BatchOutput] | None,
    ) -> Dict[str, BatchOutput]:
        """Normalize various output container shapes into a string-keyed dict."""
        if outputs is None:
            return {}
        if isinstance(outputs, Mapping):
            return {str(k): v for k, v in outputs.items() if v is not None}
        if isinstance(outputs, (list, tuple)):
            return {f"sae{i}": v for i, v in enumerate(outputs) if v is not None}
        return {"sae0": outputs}

    def _select_outputs_for_loss(
        self, outputs_map: Mapping[str, BatchOutput]
    ) -> Tuple[BatchOutput, BatchOutput]:
        """Pick obj/ano outputs with graceful fallback to the first entry."""
        if not outputs_map:
            raise ValueError("outputs_map is empty")
        outputs_obj = outputs_map.get("obj")
        outputs_ano = outputs_map.get("ano")
        first = next(iter(outputs_map.values()))
        if outputs_obj is None:
            outputs_obj = first
        if outputs_ano is None:
            outputs_ano = first
        return outputs_obj, outputs_ano

    def reset_metrics(self) -> None:
        """Reset all metric accumulators (keys are created lazily on update)."""
        self._metrics = {}

    def initialize(self) -> None:
        """Public entrypoint to start metric accumulation."""
        self._metrics_enabled = True
        self.reset_metrics()

    def _update_metrics_if_enabled(
        self, outputs: BatchOutput | Sequence[BatchOutput] | Mapping[str, BatchOutput] | None = None
    ) -> None:
        if not self._metrics_enabled:
            return
        self.update_metrics(outputs)

    def update_metrics(
        self, outputs: BatchOutput | Sequence[BatchOutput] | Mapping[str, BatchOutput] | None = None
    ) -> None:
        """Accumulate metrics for each SAE."""
        normalized = self._normalize_outputs_for_metrics(outputs)
        for role, out in normalized.items():
            if out is None:
                continue
            if role not in self._metrics:
                self._metrics[role] = {
                    name: builder() for name, builder in self._metric_builders.items()
                }
            for acc in self._metrics.get(role, {}).values():
                try:
                    acc.update(out)
                except Exception as e:
                    log.warning("Failed to update metric for %s: %s", role, e)

    def finalize(self) -> Dict[str, object]:
        """Compute aggregated metrics; returns flat dict (e.g., obj/r2, ano/mse)."""
        self._metrics_enabled = False
        summary: Dict[str, object] = {}

        def _flatten(base_key: str, val: object) -> None:
            if isinstance(val, dict):
                for k, v in val.items():
                    _flatten(f"{base_key}/{k}", v)
            elif isinstance(val, (list, tuple)):
                summary[base_key] = list(val)
            else:
                try:
                    summary[base_key] = float(val)  # type: ignore[arg-type]
                except Exception:
                    summary[base_key] = val

        for role, metrics in self._metrics.items():
            for name, acc in metrics.items():
                try:
                    val = acc.finalize()
                except Exception as e:
                    log.warning("Failed to finalize metric %s/%s: %s", role, name, e)
                    val = float("nan")
                _flatten(f"{role}/{name}", val)
        return summary
    
    def _make_patch_gt(self, gt: torch.Tensor, num_patches: int) -> torch.Tensor:
        from utils import make_patch_gt

        assert gt.ndim == 3, f"{gt.ndim} != 3"
        B, H, W = gt.shape
        target_h = (H // num_patches) * num_patches if H >= num_patches else num_patches
        target_w = (W // num_patches) * num_patches if W >= num_patches else num_patches
        resized = (target_h != H) or (target_w != W)

        if resized and (not getattr(self, "_warned_patch_gt_resize", False)):
            log.warning(
                "gt size (%d, %d) is not divisible by num_patches=%d. "
                "Resizing to (%d, %d).",
                H, W, num_patches, target_h, target_w,
            )
            self._warned_patch_gt_resize = True

        patch_gt = make_patch_gt(gt, num_patches)
        return patch_gt
    
    def _filter_center_by_neighbors(
        self, imgs: torch.Tensor, eps: float = 1e-4, mode: str = "8"
    ) -> torch.Tensor:
        """
        Suppress isolated pixels/tokens whose neighbors are all near zero.

        Args:
            imgs:
                - (B, T, C): token-wise latent. T must be a perfect square (patch grid).
                - (B, H, W, C): per-channel spatial map.
            eps:
                Threshold below which values are considered \"near zero\".
            mode:
                - \"8\": center is zeroed when all 8 neighbors are below eps.
                - \"4\": center is zeroed when 4-connected neighbors are below eps.
        """
        if imgs.ndim == 3:  # (B, T, C) tokens -> (B, H, W, C)
            B, T, C = imgs.shape
            side = int(math.isqrt(T))
            if side * side != T:
                raise ValueError(f"imgs with shape (B, T, C) requires T to be a perfect square, got T={T}")
            imgs_4d = imgs.reshape(B, side, side, C)
            reshaped_from_tokens = True
        elif imgs.ndim == 4:  # (B, H, W, C)
            imgs_4d = imgs
            B, _, _, _ = imgs_4d.shape
            reshaped_from_tokens = False
        else:
            raise ValueError("imgs must be (B, T, C) or (B, H, W, C)")

        imgs_ch_first = imgs_4d.permute(0, 3, 1, 2)  # (B, C, H, W)
        imgs_p = F.pad(imgs_ch_first, (1, 1, 1, 1), mode="constant", value=0)
        result = imgs_ch_first.clone()

        if mode == "8":
            neighbor_mask = torch.ones((3, 3), dtype=torch.bool, device=imgs.device)
            neighbor_mask[1, 1] = False  # exclude center
        elif mode == "4":
            neighbor_mask = torch.zeros((3, 3), dtype=torch.bool, device=imgs.device)
            neighbor_mask[0, 1] = True  # up
            neighbor_mask[1, 0] = True  # left
            neighbor_mask[1, 2] = True  # right
            neighbor_mask[2, 1] = True  # down
        else:
            raise ValueError("mode must be '8' or '4'")

        patches = imgs_p.unfold(2, 3, 1).unfold(3, 3, 1)  # (B, C, H, W, 3, 3)
        neighbor_vals = patches[..., neighbor_mask]  # (B, C, H, W, 8) or (B, C, H, W, 4)
        neighbor_ok = (neighbor_vals.abs() < eps).all(-1)  # True positions will be zeroed
        result[neighbor_ok] = 0.0

        result = result.permute(0, 2, 3, 1)  # (B, H, W, C)
        if reshaped_from_tokens:
            return result.reshape(B, T, C)
        return result

    def _difference_of_gaussians(
        self,
        imgs: torch.Tensor,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        kernel_size: int = 5,
    ) -> torch.Tensor:
        """
        Apply Difference of Gaussians (DoG) per channel and restore original shape.

        Args:
            imgs:
                - (B, T, C): token-wise latent. T must be a perfect square (patch grid).
                - (B, H, W, C): per-channel spatial map.
        """
        if imgs.ndim == 3:  # (B, T, C) tokens -> (B, H, W, C)
            B, T, C = imgs.shape
            side = int(math.isqrt(T))
            if side * side != T:
                raise ValueError(f"imgs with shape (B, T, C) requires T to be a perfect square, got T={T}")
            imgs_4d = imgs.reshape(B, side, side, C)
            reshaped_from_tokens = True
        elif imgs.ndim == 4:  # (B, H, W, C)
            imgs_4d = imgs
            B, _, _, _ = imgs_4d.shape
            reshaped_from_tokens = False
        else:
            raise ValueError("imgs must be (B, T, C) or (B, H, W, C)")

        # gaussian_blur expects CHW layout.
        imgs_ch_first = imgs_4d.permute(0, 3, 1, 2)  # (B, C, H, W)

        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = (k_h, k_w)
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            k_h, k_w = kernel_size
            k_h = k_h if k_h % 2 == 1 else k_h + 1
            k_w = k_w if k_w % 2 == 1 else k_w + 1
            kernel = (k_h, k_w)
        else:
            raise ValueError("kernel_size must be int or tuple/list of length 2")

        sigma1_pair = (sigma1, sigma1)
        sigma2_pair = (sigma2, sigma2)

        try:
            blur1 = F.gaussian_blur(imgs_ch_first, kernel, sigma=sigma1_pair)  # torch>=2.1
            blur2 = F.gaussian_blur(imgs_ch_first, kernel, sigma=sigma2_pair)
        except AttributeError:
            # Use the local implementation for torch<2.1.
            blur1 = self._gaussian_blur_chw(imgs_ch_first, kernel, sigma1_pair)
            blur2 = self._gaussian_blur_chw(imgs_ch_first, kernel, sigma2_pair)
        dog = blur1 - blur2

        dog = dog.permute(0, 2, 3, 1)  # (B, H, W, C)
        if reshaped_from_tokens:
            return dog.reshape(B, T, C)
        return dog

    def _gaussian_blur_chw(
        self,
        imgs_ch_first: torch.Tensor,
        kernel: Tuple[int, int],
        sigma: Tuple[float, float],
    ) -> torch.Tensor:
        """
        Fallback implementation of gaussian_blur for environments without
        torchvision / torch>=2.1.

        Args:
            imgs_ch_first: (B, C, H, W)
            kernel: (k_h, k_w)
            sigma: (sig_h, sig_w)
        """
        k_h, k_w = kernel
        sig_h, sig_w = sigma
        device = imgs_ch_first.device
        dtype = imgs_ch_first.dtype

        def _gaussian_1d(size: int, sigma_val: float) -> torch.Tensor:
            coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
            gauss = torch.exp(-0.5 * (coords / sigma_val) ** 2)
            gauss = gauss / gauss.sum()
            return gauss

        g_h = _gaussian_1d(k_h, sig_h)
        g_w = _gaussian_1d(k_w, sig_w)
        kernel_2d = torch.einsum("i,j->ij", g_h, g_w)  # (k_h, k_w)
        kernel_2d = kernel_2d.expand(imgs_ch_first.shape[1], 1, k_h, k_w)  # (C,1,k_h,k_w)

        padding = (k_w // 2, k_h // 2)
        return F.conv2d(
            imgs_ch_first,
            kernel_2d,
            bias=None,
            stride=1,
            padding=padding,
            groups=imgs_ch_first.shape[1],
        )

