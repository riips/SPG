"""guide_sae_pipeline.py
Pipeline that loads a pretrained SAE and learns guide codes on top of it
for anomaly detection (Guide SAE).

Input: ImageRecord; Output: context + Guide SAE–based anomaly scores/maps.
"""

import logging
import math
import os
from typing import List, Tuple, Dict

import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import ModelCfg, WeightModuleCfg, SAECfg
from data import ImageRecord
from encoders.base_encoder import BaseEncoder
from modules import WeightModule

from modules.sae import create_sae, BatchOutput

from .base_pipeline import BasePipeline
from .sae_pipeline import SAEPipeline
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

class GuideTensor(nn.Module):
    def __init__(self, num_guides: int, hidden_dim: int, force_nonneg: bool = False):
        super().__init__()
        self._guide_tensors = nn.Parameter(torch.randn(num_guides, hidden_dim) * 0.01 + 0.1)
        # nn.init.normal_(self._guide_tensors, std=0.01)
        self.force_nonneg = force_nonneg

    @property
    def guide_tensors(self) -> torch.Tensor:
        if self.force_nonneg:
            return F.relu(self._guide_tensors)
        else:
            return self._guide_tensors

class GuideSAEPipeline(SAEPipeline):
    """Pipeline that leverages a guide SAE for anomaly detection."""

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

        self.sae_cfg = sae_cfg
        self.sae_names = [
            cfg.name if getattr(cfg, "name", None) else f"sae{i}"
            for i, cfg in enumerate(sae_cfg)
        ]
        self.sae_list = [create_sae(sae_cfg[i], encoder.output_dim) for i in range(len(sae_cfg))]
        if len(self.sae_list) != len(self.cfg.features_list):
            raise ValueError(
                "len(method_config.sae) must match len(model.features_list). "
                f"got {len(self.sae_list)} and {len(self.cfg.features_list)}"
            )

        guide_sae_cfg = getattr(model_cfg.method_config, "guide_sae", None)
        if guide_sae_cfg is None:
            raise ValueError("guide_sae_cfg is required for GuideSAEPipeline")
        self.guide_sae_cfg = guide_sae_cfg

        run_dir = f"{guide_sae_cfg.datetime}_{guide_sae_cfg.run_tag}_{guide_sae_cfg.model_name}"
        ckpt_path = (
            f"{guide_sae_cfg.outputs_dir}/{run_dir}/checkpoints/epoch_{guide_sae_cfg.checkpoint_epoch}.pth"
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        if "sae_state_dict" not in ckpt:
            raise KeyError(f"'sae_state_dict' not found in checkpoint: {ckpt_path}")
        self._load_sae_state_dict_compat(ckpt["sae_state_dict"])
        for sae in self.sae_list:
            sae.to(device)
            sae.eval()

        # dead atom filtering mask
        self.dead_atom_masks: Dict[str, torch.Tensor] = {}
        if guide_sae_cfg.apply_dead_mask:
            for name, cfg in zip(self.sae_names, self.sae_cfg):
                self.dead_atom_masks[name] = self._load_dead_atom_mask_with_fallback(
                    outputs_dir=guide_sae_cfg.outputs_dir,
                    run_dir=run_dir,
                    sae_name=name,
                    hidden_dim=cfg.hidden_dim,
                    device=device,
                )
        else:
            for name, cfg in zip(self.sae_names, self.sae_cfg):
                self.dead_atom_masks[name] = torch.zeros(
                    cfg.hidden_dim, dtype=torch.bool, device=self.device
                )
        #####################

        self.guide_codes_list = nn.ModuleList(
            [
                GuideTensor(4, cfg.hidden_dim, force_nonneg=True)
                for cfg in self.sae_cfg
            ]
        )

        self.cur_iter = 0

        if model_cfg.optional_modules and model_cfg.optional_modules.weight_module:
            weight_cfg: WeightModuleCfg = model_cfg.optional_modules.weight_module
            self.weight_module = WeightModule(weight_cfg, encoder)
        else:
            self.weight_module = None

        self._modules = nn.ModuleDict(
            {f"guide_{i}": guide for i, guide in enumerate(self.guide_codes_list)}
        )
        if self.weight_module is not None:
            self._modules["weight_module"] = self.weight_module

        # Metric registry (extensible) and state buffers
        self._metric_builders: Dict[str, callable] = {}
        self._metrics: Dict[str, Dict[str, object]] = {"obj": {}, "ano": {}}
        self._metrics_enabled: bool = False
        self._register_builtin_metrics()
        self.reset_metrics()

    def _load_sae_state_dict_compat(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load SAE checkpoint in both old(single) and new(multi) formats."""
        has_multi_prefix = any(k.startswith("sae_list.") for k in state_dict.keys())
        if has_multi_prefix:
            for i, sae in enumerate(self.sae_list):
                prefix = f"sae_list.{i}."
                sub_state = {
                    k[len(prefix):]: v
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
                if not sub_state:
                    raise ValueError(
                        f"Checkpoint is missing parameters for sae_list[{i}] (prefix: {prefix})."
                    )
                missing, unexpected = sae.load_state_dict(sub_state, strict=False)
                if missing or unexpected:
                    log.warning(
                        "SAE[%d] load_state_dict mismatch: missing=%s unexpected=%s",
                        i,
                        missing,
                        unexpected,
                    )
            return

        # old single-SAE checkpoint
        missing, unexpected = self.sae_list[0].load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            log.warning(
                "SAE[0] load_state_dict mismatch: missing=%s unexpected=%s",
                missing,
                unexpected,
            )

    def _load_dead_atom_mask_with_fallback(
        self,
        outputs_dir: str,
        run_dir: str,
        sae_name: str,
        hidden_dim: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        new_path = f"{outputs_dir}/{run_dir}/checkpoints/dead_atom_mask_{sae_name}.pt"
        old_path = f"{outputs_dir}/{run_dir}/checkpoints/dead_atom_mask.pt"
        if os.path.exists(new_path):
            mask = torch.load(new_path, map_location=device)
            return mask.to(device)
        if os.path.exists(old_path):
            mask = torch.load(old_path, map_location=device)
            return mask.to(device)
        log.warning(
            "dead atom mask not found for '%s'. fallback to zeros. checked: %s, %s",
            sae_name,
            new_path,
            old_path,
        )
        return torch.zeros(hidden_dim, dtype=torch.bool, device=device)

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

                if len(pipeline.sae_list) != len(patch_features_list):
                    raise ValueError(
                        "len(sae_list) must match len(patch_features_list). "
                        f"got {len(pipeline.sae_list)} and {len(patch_features_list)}"
                    )

                outputs_list: List[BatchOutput] = []
                guide_outputs_list: List[BatchOutput] = []
                anomaly_scores: List[torch.Tensor] = []
                anomaly_maps: List[torch.Tensor] = []

                for i, (sae, patch_features) in enumerate(
                    zip(pipeline.sae_list, patch_features_list)
                ):
                    assert patch_features.ndim == 3
                    assert patch_features.shape[2] == sae.input_dim
                    outputs: BatchOutput = sae(patch_features, sparsify_z=False)
                    outputs_list.append(outputs)

                    guide_codes = pipeline.guide_codes_list[i].guide_tensors
                    if pipeline.guide_sae_cfg.apply_dead_mask:
                        guide_codes = guide_codes.masked_fill(
                            pipeline.dead_atom_masks[pipeline.sae_names[i]], 0.0
                        )

                    gs_cfg = pipeline.guide_sae_cfg
                    if getattr(gs_cfg, "sparsity_type", "l1") == "topk":
                        k = min(getattr(gs_cfg, "guide_topk", 32), guide_codes.shape[-1])
                        vals, idx = torch.topk(guide_codes, k, dim=-1)
                        z_hard = torch.zeros_like(guide_codes)
                        z_hard.scatter_(-1, idx, vals)
                        guide_codes = guide_codes + (z_hard - guide_codes).detach()

                    guide_tensors = sae.decode(guide_codes)
                    guide_outputs = BatchOutput(
                        x=None,
                        x_hat=guide_tensors,
                        z=guide_codes,
                        recon_error=None,
                        sparsity_penalty=None,
                        sparsify_z=False,
                        aux={},
                    )
                    guide_outputs_list.append(guide_outputs)

                    B, T, D = outputs.x.shape
                    guide_xhat = guide_outputs.x_hat
                    if guide_xhat.ndim == 2:
                        G, _ = guide_xhat.shape
                        g = guide_xhat.reshape(1, 1, G, D)
                    elif guide_xhat.ndim == 3:
                        _, G, _ = guide_xhat.shape
                        g = guide_xhat.reshape(B, 1, G, D)
                    else:
                        raise ValueError(
                            f"Invalid guide_outputs.x_hat shape: {guide_xhat.shape}"
                        )
                    x = outputs.x.reshape(B, T, 1, D)
                    g_global = g[:, :, :2, :]
                    g_local = g[:, :, 2:, :]
                    similarity = F.cosine_similarity(x, g_global, dim=-1)
                    similarity_local = F.cosine_similarity(x, g_local, dim=-1)

                    gs_cfg = pipeline.cfg.method_config.guide_sae
                    agg = gs_cfg.get("detection_aggregation", "first_token")
                    if agg == "mean_tokens":
                        x_img = x[:, 1:, :, :].mean(dim=1)
                        g_global_b = g_global.expand(B, -1, -1, -1)
                        similarity_img = F.cosine_similarity(x_img, g_global_b, dim=-1)
                        anomaly_score_i = softmax_with_temperature(
                            similarity_img, temperature=pipeline.cfg.temperature
                        ).detach().cpu()[:, 0, 1]
                    else:
                        anomaly_score_i = softmax_with_temperature(
                            similarity, temperature=pipeline.cfg.temperature
                        ).detach().cpu()[:, 0, 1]
                    scores = softmax_with_temperature(
                        similarity_local, temperature=pipeline.cfg.temperature
                    ).detach().cpu()
                    score_map = get_score_map_with_target_size(
                        scores[:, 1:], pipeline.cfg.image_size
                    )
                    anomaly_map_i = (score_map[:, 1, :, :] + 1.0 - score_map[:, 0, :, :]) / 2.0
                    anomaly_map_i = torch.from_numpy(
                        gaussian_filter(
                            anomaly_map_i.detach().cpu(), sigma=pipeline.cfg.sigma
                        )
                    )  # type: ignore

                    anomaly_scores.append(anomaly_score_i)
                    anomaly_maps.append(anomaly_map_i)

                # Update metrics automatically if enabled (first SAE for compatibility)
                pipeline._update_metrics_if_enabled({"obj": outputs_list[0]})

                anomaly_score = torch.stack(anomaly_scores, dim=0).mean(dim=0)
                anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)

                if need_context:
                    context = AnomalyContext(
                        image_features=image_features,
                        patch_features_list=patch_features_list,
                        weights=weights,
                        outputs_list=outputs_list,
                        guide_outputs_list=guide_outputs_list,
                    )
                else:
                    context = AnomalyContext()  # = {}

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

                assert "outputs_list" in ctx, "missing outputs_list in context"
                assert "guide_outputs_list" in ctx, "missing guide_outputs_list in context"
                assert "weights" in ctx, "missing weights in context"

                return pipeline._compute_loss(
                    ctx["outputs_list"],
                    ctx["guide_outputs_list"],
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
        outputs_list: List[BatchOutput],
        guide_outputs_list: List[BatchOutput],
        label: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates detection/segmentation loss with SAE recon errors.

        - CLS token is weighted by image-level `label`.
        - Patch tokens are weighted by per-patch anomaly ratios from `gt`.

        Args:
            outputs: Outputs of object SAE.
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
        assert outputs_list is not None, "outputs_list is required"
        assert weights is not None, "weights is required"
        if len(outputs_list) != len(guide_outputs_list):
            raise ValueError(
                "len(outputs_list) must match len(guide_outputs_list). "
                f"got {len(outputs_list)} and {len(guide_outputs_list)}"
            )

        self.cur_iter += 1

        det_losses: List[torch.Tensor] = []
        seg_losses: List[torch.Tensor] = []
        sparsity_losses: List[torch.Tensor] = []

        for outputs, guide_outputs in zip(outputs_list, guide_outputs_list):
            inputs = outputs.x
            assert inputs.shape == outputs.x_hat.shape, f"{inputs.shape} != {outputs.x_hat.shape}"

            B, T, D = outputs.x.shape
            guide_xhat = guide_outputs.x_hat
            if guide_xhat.ndim == 2:
                G, _ = guide_xhat.shape
                g = guide_xhat.reshape(1, 1, G, D)
            elif guide_xhat.ndim == 3:
                _, G, _ = guide_xhat.shape
                g = guide_xhat.reshape(B, 1, G, D)
            else:
                raise ValueError(f"Invalid guide_outputs.x_hat shape: {guide_xhat.shape}")
            x = outputs.x.reshape(B, T, 1, D)
            g_global = g[:, :, :2, :]
            g_local = g[:, :, 2:, :]
            similarity = F.cosine_similarity(x, g_global, dim=-1)
            similarity_local = F.cosine_similarity(x, g_local, dim=-1)

            gs_cfg = self.cfg.method_config.guide_sae
            agg = gs_cfg.detection_aggregation if gs_cfg else "first_token"
            if agg == "mean_tokens":
                x_img = x[:, 1:, :, :].mean(dim=1)
                g_global_b = g_global.expand(B, -1, -1, -1)
                similarity_img = F.cosine_similarity(x_img, g_global_b, dim=-1)
                det_sim = similarity_img[:, 0, :]
            else:
                det_sim = similarity[:, 0, :]
            det_loss = loss_function("cross_entropy", det_sim, label)

            scores = softmax_with_temperature(
                similarity_local, temperature=self.cfg.temperature
            )
            score_map = get_score_map_with_target_size(scores[:, 1:], self.cfg.image_size)

            seg_loss = 0.0
            seg_loss += loss_function("focal", score_map, gt)
            seg_loss += loss_function("dice", score_map[:, 1, :, :], gt)
            seg_loss += loss_function("dice", score_map[:, 0, :, :], 1 - gt)  # type: ignore

            if getattr(gs_cfg, "sparsity_type", "l1") == "topk":
                sparsity_loss = torch.zeros_like(guide_outputs.z.norm(p=1, dim=-1))
            else:
                sparsity_loss = guide_outputs.z.norm(p=1, dim=-1)
            det_losses.append(det_loss.mean())
            seg_losses.append(seg_loss.mean())
            sparsity_losses.append(sparsity_loss.mean())

        return {
            "det_loss": torch.stack(det_losses).mean(),
            "seg_loss": torch.stack(seg_losses).mean(),
            "sparsity_loss": torch.stack(sparsity_losses).mean()
            * self.cfg.method_config.sparsity_lmbda,
        }
    
    def post_step(self) -> None:
        for sae in self.sae_list:
            sae.decoder.enforce_unit_norm()
        