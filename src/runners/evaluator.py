from logging import Logger
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd  # type: ignore
from skimage.segmentation import mark_boundaries  # type: ignore
import torch
from torch.nn.modules.container import ModuleDict
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as F  # type: ignore
from tqdm import tqdm

from configs import BaseCfg
from data import DataManager, Dataset, ImageRecord
from encoders import EncoderManager
from metrics import cal_score
from data.meta import get_specie_names, get_normal_species, has_species_info
from pipelines.base_pipeline import BasePipeline
from pipelines.pipeline_factory import get_pipeline
from pipelines.core.contracts import PipelineOutput
from cache import CacheManager, build_feature_cache
from utils import CheckpointManager

from .base_runner import BaseRunner

log: Logger = logging.getLogger(name=__name__)


def _image_level_score_from_outputs(
    anomaly_score: torch.Tensor,
    anomaly_map: Optional[torch.Tensor],
    mode: str,
    map_pool: str,
    top_q_percent: float,
    alpha: float,
    map_pool_tau: float = 1.0,
) -> torch.Tensor:
    """Compute image-level score from pipeline score and/or anomaly map (eval only)."""
    out = anomaly_score.detach().cpu().float()
    need_map = mode in ("map", "hybrid")
    if anomaly_map is None and need_map:
        return out
    if not need_map:
        return out
    m = anomaly_map.detach().cpu().float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    flat = m.flatten(-2)
    if map_pool == "max":
        s_map = flat.amax(dim=-1)
    elif map_pool == "log_sum_exp":
        tau = max(map_pool_tau, 1e-8)
        b = flat.amax(dim=-1, keepdim=True)
        inner = (torch.exp((flat - b) / tau).mean(dim=-1)).clamp(min=1e-10)
        s_map = b.squeeze(-1) + tau * torch.log(inner)
    else:
        k = max(1, int(flat.shape[-1] * (top_q_percent / 100.0)))
        topk = flat.topk(k, dim=-1).values
        s_map = topk.mean(dim=-1)
    if mode == "map":
        return s_map
    out = out.squeeze()
    if out.ndim == 0:
        out = out.unsqueeze(0)
    s_map = s_map.to(out.device)
    return (alpha * out + (1.0 - alpha) * s_map).to(out.device)


class Evaluator(BaseRunner):
    def __init__(self, cfg: BaseCfg) -> None:
        """
        Initializes the Evaluator class.

        Args:
            cfg: The evaluation configuration object.
        """
        super().__init__(cfg=cfg)

        self.cfg = cfg = self._merge_cfg(cfg)
        log.info("Config:\n%s", OmegaConf.to_yaml(self.cfg, resolve=True))

        # Load model
        self.encoder_manager = EncoderManager(model_cfg=cfg.model, device=self.device)
        self.encoder = self.encoder_manager.load_encoder().to(self.device)

        # Initialize cache manager (for feature reuse)
        cache_dir = os.environ.get("CACHE_DIR") or cfg.cache.dir
        self.cache_mgr = CacheManager(
            {
                "model_id": cfg.model.id,
                "dataset_name": cfg.test_data.dataset_name,
                "image_size": cfg.model.image_size,
                "features_list": cfg.model.features_list,
            },
            cache_dir,
        )

        # Build dataset and dataloader
        filter_kw = getattr(cfg.test_data, "filter_kw", None)
        combined_datasets = getattr(cfg.test_data, "combined_datasets", None)

        if cfg.test_data.input_type == "feature":
            _cache_manager = self.cache_mgr
            if not self.cache_mgr.exists():
                build_feature_cache(
                    self.encoder, self.device, cfg.test_data, cfg.model, self.cache_mgr
                )
        elif cfg.test_data.input_type == "image":
            _cache_manager = None
        else:
            raise ValueError(f"Invalid input type: {cfg.test_data.input_type}")

        self.data_mgr: DataManager = DataManager(
            dataset_cfg=cfg.test_data,
            model_cfg=cfg.model,
            batch_size=cfg.evaluate.batch_size,
            mode=cfg.test_data.input_type,
            cache_mgr=_cache_manager,
            filter_kw=filter_kw,
            combined_datasets=combined_datasets,
        )
        self.dataset: Dataset = self.data_mgr.dataset
        self.dataloader: DataLoader[ImageRecord] = self.data_mgr.get_dataloader(
            shuffle=False
        )
        # anomaly map saving config
        self.save_maps = bool(getattr(cfg.evaluate, "save_maps", False))
        self.map_save_dir: Optional[Path] = None
        self.map_dtype_np = np.float32
        if self.save_maps:
            map_dtype = str(getattr(cfg.evaluate, "map_dtype", "float32")).lower()
            self.map_dtype_np = np.float16 if map_dtype == "float16" else np.float32
            maps_dir_cfg = getattr(cfg.evaluate, "maps_dir", None)
            default_maps_dir = (
                Path(cfg.base_dir)
                / "artifacts"
                / self.data_mgr.dataset_name
                / f"epoch_{cfg.evaluate.epoch}"
                / "maps"
            )
            self.map_save_dir = Path(maps_dir_cfg) if maps_dir_cfg else default_maps_dir
            self.map_save_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                "Anomaly maps will be saved to %s (dtype=%s)",
                self.map_save_dir,
                self.map_dtype_np,
            )
        self.encoder.eval()

        # Build processing pipeline
        self.pipeline: BasePipeline = get_pipeline(model_cfg=cfg.model)(
            model_cfg=cfg.model, encoder=self.encoder, device=self.device
        )
        self.modules: ModuleDict = self.pipeline.get_modules()
        self.modules.eval()
        self.modules.to(self.device)

        # Load Checkpoint
        self.checkpoint_manager: CheckpointManager = CheckpointManager(
            save_dir=cfg.save_dir, device=self.device
        )
        ckpt_path = os.path.join(cfg.save_dir, cfg.evaluate.ckpt)
        use_ema = bool(getattr(cfg.evaluate, "use_ema", False))
        if use_ema:
            _, loaded = self.checkpoint_manager.load(
                path=ckpt_path,
                ema_modules_dict=self.modules,
            )
            if "ema_modules_dict" not in loaded:
                log.warning(
                    "EMA weights not found in checkpoint; falling back to base model. "
                    "Re-save checkpoints with EMA enabled to evaluate EMA weights."
                )
                self.checkpoint_manager.load(
                    path=ckpt_path,
                    modules_dict=self.modules,
                )
        else:
            self.checkpoint_manager.load(
                path=ckpt_path,
                modules_dict=self.modules,
            )

    def _merge_cfg(self, cfg: BaseCfg) -> BaseCfg:
        # Restore training-time configuration from the Hydra config under base_dir
        train_dir = Path(cfg.base_dir).expanduser().resolve()
        train_cfg_path = train_dir / ".hydra" / "config.yaml"
        train_cfg: DictConfig = OmegaConf.load(train_cfg_path)  # type: ignore

        for k in ("hydra", "mode"):
            train_cfg.pop(k, None)

        merge_keys = ("model", "base_dir", "save_dir")
        train_subset = {k: train_cfg[k] for k in merge_keys if k in train_cfg}

        OmegaConf.set_struct(cfg, False)  # Temporarily disable structured config checks. # type: ignore
        cfg = OmegaConf.merge(cfg, OmegaConf.create(train_subset))  # type: ignore
        OmegaConf.set_struct(cfg, True)  # type: ignore

        return cfg

    def evaluate(
        self,
        get_features: bool = False,
        get_similarity: bool = False,
        eval_performance: bool = True,
        aggregate_anomaly_map: bool = True,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Run evaluation on the test dataset.

        Args:
            get_features (bool): Whether to extract and store image-level features.
            get_similarity (bool): Whether to extract and store similarity maps.
            eval_performance (bool): Whether to compute and print evaluation metrics.
            aggregate_anomaly_map (bool): Whether to spatially aggregate the anomaly maps.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - results: Per-image anomaly and feature data.
                - performance: Evaluation metrics if eval_performance is True.
        """

        # Initialize a results dictionary
        results: Dict[str, Dict[str, Any]] = {}

        # Store per-image evaluation results
        records: List[Dict[str, Any]] = []

        by_specie_enabled = bool(getattr(self.cfg.evaluate, "by_specie", False))

        # Initialize SAE metrics accumulation if available
        if hasattr(self.pipeline, "initialize"):
            try:
                self.pipeline.initialize()  # type: ignore[attr-defined]
            except Exception as e:
                log.warning("Failed to initialize pipeline: %s", e)

        # Cache per-class species metadata (species list / normal species)
        class_species_list: Dict[str, List[str]] = {}
        class_normal_species: Dict[str, List[str]] = {}

        # Start measuring inference time
        start_time = time.time()
        for _, inputs in enumerate(cast(Iterable[ImageRecord], tqdm(self.dataloader))):
            cls_name = inputs.cls_name[0]
            specie = inputs.specie_name[0]

            # Initialize the class on first encounter.
            if cls_name not in results:
                results[cls_name] = {}
                for key in [
                    "gt_sp",
                    "gt_px",
                    "pr_sp",
                    "pr_px",
                    "image_features",
                    "image_similarity",
                ]:
                    results[cls_name][key] = []
                if by_specie_enabled:
                    # Nested bucket for species aggregation.
                    results[cls_name]["by_specie"] = {}

                    # Initialize species information for the class.
                    ds_name = self.data_mgr.dataset_name
                    try:
                        if has_species_info(ds_name, cls_name):
                            class_species_list[cls_name] = get_specie_names(ds_name, cls_name)
                            class_normal_species[cls_name] = get_normal_species(ds_name, cls_name)
                        else:
                            class_species_list[cls_name] = []
                            class_normal_species[cls_name] = ["good"]
                    except Exception:
                        class_species_list[cls_name] = []
                        class_normal_species[cls_name] = ["good"]

                    # Pre-create buckets only for anomaly species; normal species are excluded.
                    for sp in class_species_list[cls_name]:
                        if sp in class_normal_species[cls_name]:
                            continue
                        results[cls_name]["by_specie"][sp] = {k: [] for k in ["gt_sp", "gt_px", "pr_sp", "pr_px"]}

            # Create a bucket for an anomaly species if it does not exist yet,
            # such as species encountered in datasets without metadata.
            if (
                by_specie_enabled
                and specie not in results[cls_name]["by_specie"]
                and specie not in class_normal_species.get(cls_name, [])
            ):
                results[cls_name]["by_specie"][specie] = {k: [] for k in ["gt_sp", "gt_px", "pr_sp", "pr_px"]}

            gt_mask = inputs.img_mask
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name]["gt_px"].append(gt_mask)
            results[cls_name]["gt_sp"].append(inputs.anomaly)

            # with torch.no_grad(), torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs: PipelineOutput = self.pipeline.forward(
                    inputs,
                    get_features=get_features,
                    get_similarity=get_similarity,
                    get_loss=False,
                    get_score=True,
                    aggregate_anomaly_map=aggregate_anomaly_map,
                )
                anomaly_score = outputs.get("anomaly_score")
                anomaly_map = outputs.get("anomaly_map")

                img_score_cfg = getattr(self.cfg.evaluate, "image_score", None)
                if img_score_cfg is not None:
                    image_score_used = _image_level_score_from_outputs(
                        anomaly_score,
                        anomaly_map,
                        mode=getattr(img_score_cfg, "mode", "pipeline"),
                        map_pool=getattr(img_score_cfg, "map_pool", "max"),
                        top_q_percent=float(getattr(img_score_cfg, "top_q_percent", 10.0)),
                        alpha=float(getattr(img_score_cfg, "alpha", 0.5)),
                        map_pool_tau=float(getattr(img_score_cfg, "map_pool_tau", 1.0)),
                    )
                else:
                    image_score_used = anomaly_score.detach().cpu()

                if get_features:
                    results[cls_name]["image_features"].append(outputs.get("image_features").detach().cpu())  # type: ignore
                if get_similarity:
                    results[cls_name]["image_similarity"].append(outputs.get("image_similarity").detach().cpu())  # type: ignore

                results[cls_name]["pr_sp"].append(image_score_used)  # type: ignore
                results[cls_name]["pr_px"].append(anomaly_map.detach().cpu())  # type: ignore

                # Save anomaly map per image if enabled
                if self.save_maps and self.map_save_dir is not None and anomaly_map is not None:
                    try:
                        global_id = int(inputs.global_id.item())  # type: ignore
                    except Exception:
                        global_id = len(records)
                    try:
                        local_id = int(inputs.local_id.item())  # type: ignore
                    except Exception:
                        local_id = -1

                    try:
                        map_np = anomaly_map.detach().cpu().numpy().astype(self.map_dtype_np)
                        np.savez_compressed(
                            self.map_save_dir / f"{global_id:06d}.npz",
                            anomaly_map=map_np,
                            cls_name=cls_name,
                            specie_name=specie,
                            image_path=inputs.img_path[0],
                            global_id=global_id,
                            local_id=local_id,
                            anomaly=int(inputs.anomaly.item()),  # type: ignore
                        )
                    except Exception as e:
                        log.warning(
                            "Failed to save anomaly map for %s/%s (global_id=%s): %s",
                            cls_name,
                            specie,
                            global_id,
                            e,
                        )

                if by_specie_enabled:
                    # Also accumulate into species buckets.
                    # Rule: evaluate target species S only over {normal(good), S}.
                    normal_set = set(class_normal_species.get(cls_name, ["good"]))
                    anomaly_species = [sp for sp in results[cls_name]["by_specie"].keys()]

                    pr_score_cpu = image_score_used
                    if specie in normal_set:
                        # Add good samples as negatives to every target-species bucket.
                        for sp in anomaly_species:
                            results[cls_name]["by_specie"][sp]["gt_px"].append(gt_mask)
                            results[cls_name]["by_specie"][sp]["gt_sp"].append(torch.zeros_like(pr_score_cpu))
                            results[cls_name]["by_specie"][sp]["pr_sp"].append(pr_score_cpu)  # type: ignore
                            results[cls_name]["by_specie"][sp]["pr_px"].append(anomaly_map.detach().cpu())  # type: ignore
                    else:
                        # Add anomaly samples as positives only to their own species.
                        # Other species do not include them in their population.
                        if specie in results[cls_name]["by_specie"]:
                            results[cls_name]["by_specie"][specie]["gt_px"].append(gt_mask)
                            results[cls_name]["by_specie"][specie]["gt_sp"].append(torch.ones_like(pr_score_cpu))
                            results[cls_name]["by_specie"][specie]["pr_sp"].append(pr_score_cpu)  # type: ignore
                            results[cls_name]["by_specie"][specie]["pr_px"].append(anomaly_map.detach().cpu())  # type: ignore

            # Record evaluation result for each image
            records.append(
                {
                    "global_id": inputs.global_id.item(),  # type: ignore
                    "local_id": inputs.local_id.item(),  # type: ignore
                    "image_path": inputs.img_path[0],
                    "model": self.cfg.model.id,
                    "cls_name": inputs.cls_name[0],
                    "specie_name": inputs.specie_name[0],
                    "anomaly": inputs.anomaly.item(),  # type: ignore
                    "anomaly_score": float(image_score_used.detach().cpu().item()),  # type: ignore
                }
            )

        # End measuring inference time
        end_time = time.time()
        log.info(f"Total Inference Time: {(end_time - start_time) * 1000:.3f} [ms]")
        log.info(
            f"Throughput: {len(self.dataloader) / (end_time - start_time):.3f} [img/s]"
        )

        # Summarize SAE metrics if available
        if hasattr(self.pipeline, "finalize"):
            try:
                metrics_summary = self.pipeline.finalize()  # type: ignore[attr-defined]
                log.info("SAE metrics summary: %s", metrics_summary)
            except Exception as e:
                log.warning("Failed to finalize pipeline: %s", e)

        # Evaluate overall performance (if enabled)
        performance: Dict[str, Dict[str, Any]] = {}
        if eval_performance:
            performance_cls, performance_sp = self.metric(results)
            # Class-wise CSV
            self._show_performance(performance_cls)
            # Additional: species-wise CSV
            if by_specie_enabled:
                self._show_species_performance(performance_sp)
            performance = performance_cls

        # Visualize prediction results (if enabled)
        try:
            vis_enabled = hasattr(self.cfg, "visualize") and getattr(
                self.cfg.visualize, "enabled", False
            )
        except Exception:
            vis_enabled = False
        if vis_enabled:
            self._visualize(results)

        # Save per-image evaluation records
        df_img = pd.DataFrame(records)
        filter_suffix = self._get_filter_suffix()
        path = (
            Path(self.cfg.base_dir)
            / "sample-wise_results"
            / f"image_metrics_{self.data_mgr.dataset_name}_{self.cfg.evaluate.epoch}{filter_suffix}.parquet"
        )
        os.makedirs(path.parent, exist_ok=True)

        # Save filter conditions into parquet metadata (if present)
        filter_kw = getattr(self.cfg.test_data, "filter_kw", None)
        if filter_kw:
            # Save filter conditions in DataFrame metadata.
            df_img.attrs["filter_conditions"] = str(filter_kw)

        df_img.to_parquet(path)  # type: ignore
        log.info(f"Saved per-image metrics → {path}")
        if self.save_maps and self.map_save_dir is not None:
            log.info(
                "Saved anomaly maps to %s (count=%d)",
                self.map_save_dir,
                len(records),
            )

        return results, performance

    def metric(self, results: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
        """
        Calculates image-level and pixel-level evaluation metrics.

        Args:
            results (Dict): Dictionary containing prediction and ground truth for each image.

        Returns:
            Tuple[Dict, Dict]: (class_performance, species_performance)
        """
        cfg = self.cfg.evaluate
        by_specie_enabled = bool(getattr(cfg, "by_specie", False))
        performance_cls: Dict[str, Dict[str, Any]] = {}
        performance_sp: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for cls_name, bucket in results.items():
            # Class-wise metrics.
            performance_cls[cls_name] = {}

            if cfg.metrics.image_level is not None and len(bucket.get("pr_sp", [])) > 0:
                pr = np.concatenate(bucket["pr_sp"])
                gt = np.concatenate(bucket["gt_sp"])
                for _metric in cfg.metrics.image_level:
                    _score, _ = cal_score(
                        gt, pr, _metric,
                        pro_use_fast=getattr(self.cfg.evaluate, "pro_use_fast", False),
                    )
                    performance_cls[cls_name][f"image-{_metric}"] = _score
                    log.debug(
                        f"{cls_name} image {_metric:5} {performance_cls[cls_name][f'image-{_metric}']}"
                    )

            if cfg.metrics.pixel_level is not None and len(bucket.get("pr_px", [])) > 0:
                pr = np.concatenate(bucket["pr_px"])
                gt = np.concatenate(bucket["gt_px"])
                for _metric in cfg.metrics.pixel_level:
                    _score, _ = cal_score(
                        gt, pr, _metric,
                        pro_use_fast=getattr(self.cfg.evaluate, "pro_use_fast", False),
                    )
                    performance_cls[cls_name][f"pixel-{_metric}"] = _score
                    log.debug(
                        f"{cls_name} pixel {_metric:5} {performance_cls[cls_name][f'pixel-{_metric}']}"
                    )

            # Species-wise metrics.
            if (
                by_specie_enabled
                and "by_specie" in bucket
                and isinstance(bucket["by_specie"], dict)
            ):
                performance_sp[cls_name] = {}
                for specie, sb in bucket["by_specie"].items():
                    rec: Dict[str, Any] = {}
                    # Image level.
                    if cfg.metrics.image_level is not None and len(sb.get("pr_sp", [])) > 0:
                        pr_list = sb["pr_sp"]
                        gt_list = sb["gt_sp"]
                        try:
                            pr = np.concatenate(pr_list)
                            gt = np.concatenate(gt_list)
                            # Record image counts (normal/anomaly).
                            gt_flat = gt.ravel()
                            num_anom = int(gt_flat.sum())
                            num_norm = int(gt_flat.size - num_anom)
                            rec["num_normal"] = num_norm
                            rec["num_anomaly"] = num_anom
                            if gt.max() == gt.min():
                                # If only one side is present, output 0.0 for all metrics.
                                for _metric in cfg.metrics.image_level:
                                    rec[f"image-{_metric}"] = 0.0
                            else:
                                for _metric in cfg.metrics.image_level:
                                    _score, _ = cal_score(
                                        gt, pr, _metric,
                                        pro_use_fast=getattr(self.cfg.evaluate, "pro_use_fast", False),
                                    )
                                    rec[f"image-{_metric}"] = _score
                        except Exception:
                            # Fill with 0.0 on errors.
                            rec.setdefault("num_normal", 0)
                            rec.setdefault("num_anomaly", 0)
                            for _metric in cfg.metrics.image_level:
                                rec[f"image-{_metric}"] = 0.0
                    # Pixel level.
                    if cfg.metrics.pixel_level is not None and len(sb.get("pr_px", [])) > 0:
                        pr_list = sb["pr_px"]
                        gt_list = sb["gt_px"]
                        try:
                            pr = np.concatenate(pr_list)
                            gt = np.concatenate(gt_list)
                            if gt.max() == gt.min():
                                for _metric in cfg.metrics.pixel_level:
                                    rec[f"pixel-{_metric}"] = 0.0
                            else:
                                for _metric in cfg.metrics.pixel_level:
                                    _score, _ = cal_score(
                                        gt, pr, _metric,
                                        pro_use_fast=getattr(self.cfg.evaluate, "pro_use_fast", False),
                                    )
                                    rec[f"pixel-{_metric}"] = _score
                        except Exception:
                            for _metric in cfg.metrics.pixel_level:
                                rec[f"pixel-{_metric}"] = 0.0

                    # Save the zero-filled record if rows should be preserved even when empty.
                    performance_sp[cls_name][specie] = rec

        return performance_cls, performance_sp

    def _show_species_performance(self, data_sp: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Logs and saves species-level evaluation metrics to a CSV file.

        Args:
            data_sp (Dict): {class: {specie: metrics}}
        """
        # Skip if there is no species-level data
        if not data_sp:
            return

        rows: List[Dict[str, Any]] = []
        for cls_name, sp_dict in data_sp.items():
            if not sp_dict:
                continue
            for specie, metrics in sp_dict.items():
                row = {"class": cls_name, "specie": specie}
                row.update(metrics)
                rows.append(row)

        if not rows:
            return

        df = pd.DataFrame.from_records(rows)  # type: ignore
        # Fill NaNs with 0.0
        df = df.fillna(0.0)
        # Do not add extra mean rows here; just sort and save
        df = df.sort_values(by=["class", "specie"]).reset_index(drop=True)

        filter_suffix = self._get_filter_suffix()
        csv_path = (
            Path(self.cfg.base_dir)
            / "metrics"
            / f"metrics_{self.data_mgr.dataset_name}_{self.cfg.evaluate.epoch}{filter_suffix}_by-specie.csv"
        )
        os.makedirs(csv_path.parent, exist_ok=True)

        filter_kw = getattr(self.cfg.test_data, "filter_kw", None)
        if filter_kw:
            filter_info = f"# Filter conditions: {filter_kw}\n"
            with open(csv_path, "w") as f:
                f.write(filter_info)
            df.to_csv(csv_path, mode="a", index=False)  # type: ignore
        else:
            df.to_csv(csv_path, index=False)  # type: ignore

        log.info(f"Saved species metrics to {csv_path}")
        print(df.to_markdown(index=False, floatfmt=".3f"))  # type: ignore

    def _show_performance(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Logs and saves evaluation metrics to a CSV file.

        Args:
            data (Dict): Dictionary containing performance results.
        """
        records: List[Dict[str, Any]] = [
            {"class": cls, **metrics} for cls, metrics in data.items()
        ]
        df = pd.DataFrame.from_records(records)  # type: ignore

        mean_row = df.mean(numeric_only=True)  # type: ignore
        mean_row["class"] = "Mean"  # type: ignore
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        # Generate filename suffix based on filter conditions
        filter_suffix = self._get_filter_suffix()
        csv_path = (
            Path(self.cfg.base_dir)
            / "metrics"
            / f"metrics_{self.data_mgr.dataset_name}_{self.cfg.evaluate.epoch}{filter_suffix}.csv"
        )
        os.makedirs(csv_path.parent, exist_ok=True)

        # Optionally write filter conditions as a comment at the top of the CSV
        filter_kw = getattr(self.cfg.test_data, "filter_kw", None)
        if filter_kw:
            # Add filter conditions as a comment at the top of the CSV file.
            filter_info = f"# Filter conditions: {filter_kw}\n"
            with open(csv_path, "w") as f:
                f.write(filter_info)
            df.to_csv(csv_path, mode="a", index=False)  # type: ignore
        else:
            df.to_csv(csv_path, index=False)  # type: ignore

        log.info(f"Saved metrics to {csv_path}")
        print(df.to_markdown(index=False, floatfmt=".3f"))  # type: ignore

    def _visualize(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Visualize input images and anomaly maps and save them to disk.

        Args:
            results (Dict): Dictionary containing pixel-level predictions.
        """
        base_dir = (
            Path(self.cfg.base_dir) / "visualization" / self.data_mgr.dataset_name
        )
        vis_cfg = getattr(self.cfg, "visualize", None)
        alpha = 0.5
        boundary_color = (1.0, 0.0, 0.0)
        boundary_mode = "thick"
        save_inputs = True
        save_anomaly = True
        save_overlay = True
        if vis_cfg is not None:
            alpha = float(getattr(vis_cfg, "alpha", alpha))
            try:
                color_list = getattr(vis_cfg, "boundary_color", list(boundary_color))
                boundary_color = tuple(float(c) for c in color_list)
            except Exception:
                pass
            boundary_mode = getattr(vis_cfg, "boundary_mode", boundary_mode)
            save_inputs = bool(getattr(vis_cfg, "save_inputs", save_inputs))
            save_anomaly = bool(getattr(vis_cfg, "save_anomaly", save_anomaly))
            save_overlay = bool(getattr(vis_cfg, "save_overlay", save_overlay))

        # Use the filtered dataset (`self.dataset`) for visualization
        for cls_name in results.keys():
            save_dir = base_dir / cls_name
            os.makedirs(save_dir, exist_ok=True)
            pr_px_list = results[cls_name]["pr_px"]
            # log.info(f"pr_px_list: {len(pr_px_list)}, {pr_px_list[0].shape}")
            vmax = torch.cat(pr_px_list).max().item()
            vmin = torch.cat(pr_px_list).min().item()

            # Get per-class thresholds from metrics.cal_score; default metric is f1.
            best_thr = None
            try:
                gt_px_list = results[cls_name]["gt_px"]
                pr_concat = np.concatenate(
                    [p.flatten() for p in torch.cat(pr_px_list).detach().cpu().numpy()]
                )
                gt_concat = np.concatenate(
                    [g.flatten() for g in torch.cat(gt_px_list).detach().cpu().numpy()]
                )

                # Select the metric used for visualization.
                metric_name = "f1"
                if vis_cfg is not None:
                    if getattr(vis_cfg, "metric", None):
                        metric_name = vis_cfg.metric

                _score, thr = cal_score(
                    gt_concat, pr_concat, metric_name,
                    pro_use_fast=getattr(self.cfg.evaluate, "pro_use_fast", False),
                )
                if thr is not None:
                    best_thr = float(thr)
                    log.info(
                        f"Class {cls_name}: {metric_name}={_score:.4f} at thr={best_thr:.6f}"
                    )
                else:
                    log.warning(
                        f"{cls_name}: metric {metric_name} では閾値が定義されないため輪郭描画用閾値は使用しません"
                    )
            except Exception as e:
                log.warning(f"{cls_name}: 閾値計算をスキップします ({e})")
                best_thr = None

            # Get indices for this class from the filtered dataset.
            cls_indices: List[int] = []
            for idx in range(len(self.dataset)):
                try:
                    # Retrieve data directly from the filtered dataset.
                    record = self.dataset[idx]
                    if hasattr(record, "cls_name") and record.cls_name == cls_name:
                        cls_indices.append(idx)
                except Exception as e:
                    log.warning(
                        f"データセットからインデックス {idx} のデータを取得できませんでした: {e}"
                    )
                    continue

            # Ensure the number of predictions matches the number of images for this class
            if len(pr_px_list) != len(cls_indices):
                log.warning(
                    f"{cls_name} の予測数({len(pr_px_list)})と画像数({len(cls_indices)})が一致しません。"
                )
                log.warning(
                    f"フィルタリング条件: {getattr(self.cfg.test_data, 'filter_kw', 'なし')}"
                )
                # Align to the shorter length.
                min_len = min(len(pr_px_list), len(cls_indices))
                pr_px_list = pr_px_list[:min_len]
                cls_indices = cls_indices[:min_len]

            for i, pred in enumerate(pr_px_list):
                if i >= len(cls_indices):
                    break
                img_idx = cls_indices[i]

                try:
                    # Retrieve the image from the filtered dataset
                    record = self.dataset[img_idx]

                    # If running in feature mode, reload the image from disk
                    if record.img is None:
                        # Load image from file
                        from PIL import Image
                        import torchvision.transforms as transforms  # type: ignore

                        img_path = Path(self.cfg.test_data.root) / record.img_path
                        if not img_path.exists():
                            log.error(f"画像ファイルが見つかりません: {img_path}")
                            continue

                        # Read image
                        img = Image.open(img_path).convert("RGB")

                        # Apply preprocessing (resize and ToTensor)
                        if (
                            hasattr(self.data_mgr, "preprocess")
                            and self.data_mgr.preprocess is not None
                        ):
                            img = self.data_mgr.preprocess(img)  # type: ignore
                        else:
                            # Default preprocessing using cfg.model.image_size
                            target_size = 224
                            try:
                                target_size = int(
                                    getattr(self.cfg.model, "image_size", target_size)
                                )
                            except Exception:
                                pass
                            transform = transforms.Compose(
                                [
                                    transforms.Resize((target_size, target_size)),
                                    transforms.ToTensor(),
                                ]
                            )
                            img = transform(img)  # type: ignore
                    else:
                        img = record.img  # type: ignore

                    # img: torch.Tensor (C, H, W) or (H, W, C); convert to numpy for visualization
                    if isinstance(img, torch.Tensor):
                        img_np = img.detach().cpu().numpy()  # type: ignore
                    else:
                        img_np = np.array(img)  # type: ignore

                    # (C, H, W) -> (H, W, C) for 3-channel images
                    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                        img_np = np.transpose(img_np, (1, 2, 0))  # type: ignore

                    # Normalize image to [0, 1] depending on its current range
                    if img_np is not None:  # type: ignore
                        img_min, img_max = img_np.min(), img_np.max()

                        # Case 1: standardized image (roughly [-3, 3])
                        if img_min < -0.5 and img_max > 0.5:
                            img_np = (img_np - img_min) / (img_max - img_min + 1e-8)
                        # Case 2: raw 0-255 image
                        elif img_min >= 0 and img_max > 100:
                            img_np = img_np / 255.0
                        # Case 3: already in [0, 1]
                        elif img_min >= 0 and img_max <= 1:
                            pass  # Already normalized.
                        # Case 4: fallback to min-max normalization
                        else:
                            img_np = (img_np - img_min) / (img_max - img_min + 1e-8)

                    # Convert anomaly map to numpy
                    pred_np = pred.squeeze().detach().cpu().numpy()

                    # Map anomaly scores to an RGB image using a colormap
                    anomaly_map_rgb = plt.cm.jet(  # type: ignore
                        (pred_np - vmin) / (vmax - vmin + 1e-8)
                    )[
                        :, :, :3
                    ]  # (H, W, 3)

                    # If the input is grayscale, convert it to RGB
                    if img_np.ndim == 2:
                        img_np = np.stack([img_np] * 3, axis=-1)  # type: ignore
                    elif img_np.shape[2] == 1:
                        img_np = np.concatenate([img_np] * 3, axis=-1)  # type: ignore

                    # Resize anomaly map to match the input image size (for visualization)
                    target_h, target_w = img_np.shape[:2]
                    if (target_h, target_w) != anomaly_map_rgb.shape[:2]:  # type: ignore
                        from PIL import Image

                        # Resize anomaly map RGB image to the input size
                        anomaly_map_pil = Image.fromarray((anomaly_map_rgb * 255).astype(np.uint8))  # type: ignore
                        anomaly_map_resized = F.resize(anomaly_map_pil, (target_h, target_w), antialias=True)  # type: ignore
                        anomaly_map_rgb = np.array(anomaly_map_resized) / 255.0

                    # Resize raw score map for contour drawing
                    if pred_np.shape[:2] != (target_h, target_w):
                        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float()  # type: ignore
                        pred_resized = F.resize(pred_tensor, [target_h, target_w], antialias=True).squeeze().numpy()  # type: ignore
                    else:
                        pred_resized = pred_np

                    # Alpha-blend the input image and anomaly map RGB image
                    overlay = (1 - alpha) * img_np + alpha * anomaly_map_rgb  # type: ignore
                    overlay = np.clip(overlay, 0, 1)  # type: ignore

                    # If a threshold is available, draw contours for anomaly regions
                    if best_thr is not None:
                        try:
                            binary_mask = (pred_resized >= best_thr).astype(np.bool_)  # type: ignore
                            overlay = mark_boundaries(overlay, binary_mask, color=boundary_color, mode=boundary_mode)  # type: ignore
                        except Exception as ee:
                            log.warning(
                                "Failed to draw contour for class %s image %d: %s",
                                cls_name,
                                i,
                                ee,
                            )

                    # Save overlay/input/anomaly images depending on flags
                    if save_overlay:
                        plt.imsave(save_dir / f"{i:03d}_overlay.png", overlay)  # type: ignore
                    if save_inputs:
                        plt.imsave(save_dir / f"{i:03d}_input.png", img_np)  # type: ignore
                    if save_anomaly:
                        plt.imsave(save_dir / f"{i:03d}_anomaly.png", anomaly_map_rgb)  # type: ignore

                except Exception as e:
                    log.error("Error during visualization for image %d: %s", i, e)
                    continue

        log.info(f"Visualization finished: {base_dir}")

    def _get_filter_suffix(self) -> str:
        """
        Generate a suffix for CSV filenames based on filter conditions.

        Returns:
            str: Suffix string for the filename.
        """
        filter_kw = getattr(self.cfg.test_data, "filter_kw", None)
        suffix_parts: List[str] = []
        if filter_kw:
            # Class-name filter
            if "cls_names" in filter_kw and filter_kw["cls_names"]:
                cls_names = filter_kw["cls_names"]
                if len(cls_names) == 1:
                    suffix_parts.append(f"_cls_{cls_names[0]}")  # type: ignore
                else:
                    suffix_parts.append(f"_cls_{'-'.join(cls_names)}")  # type: ignore

            # Anomaly/normal label filter
            if "anomaly" in filter_kw and filter_kw["anomaly"] is not None:
                anomaly_label = "normal" if filter_kw["anomaly"] == 0 else "anomaly"
                suffix_parts.append(f"_{anomaly_label}")  # type: ignore

            # Species-name filter
            if "specie_names" in filter_kw and filter_kw["specie_names"]:
                specie_names = filter_kw["specie_names"]
                if len(specie_names) == 1:
                    suffix_parts.append(f"_specie_{specie_names[0]}")  # type: ignore
                else:
                    suffix_parts.append(f"_specie_{'-'.join(specie_names)}")  # type: ignore

        if bool(getattr(self.cfg.evaluate, "use_ema", False)):
            suffix_parts.append("_ema")

        return "".join(suffix_parts)  # type: ignore
