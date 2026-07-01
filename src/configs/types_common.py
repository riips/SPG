"""Common configuration dataclasses shared across pipelines."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class WeightModuleCfg:
    dynamic: bool = False
    """Use image-conditioned weights instead of fixed ones."""
    num_layers: int = 4
    """Number of feature layers to weight."""
    use_bias: bool = True
    """Whether to add bias in the weighting MLP."""

@dataclass
class GuideSAECfg:
    """Guide SAE specific configuration."""

    datetime: str
    """Run datetime used in output path (e.g. '2026-02-03/20-36-06')."""

    model_name: str
    """Encoder model name used in output path."""

    run_tag: str = "none"
    """Tag used in output dir name between datetime and model name."""

    outputs_dir: str = "${REPO_ROOT}/outputs"
    """Base outputs directory containing checkpoints (override to match your environment)."""

    checkpoint_epoch: int = 50
    """Checkpoint epoch index used for SAE state dict."""

    apply_dead_mask: bool = True
    """Whether to apply dead atom mask."""

    detection_aggregation: Literal["first_token", "mean_tokens"] = "first_token"
    """How to aggregate tokens for image-level detection: first token only, or mean of non-first tokens."""

    sparsity_type: Literal["l1", "topk"] = "l1"
    """Sparsification for guide codes: 'l1' (L1 penalty only) or 'topk' (keep top-k per guide vector)."""

    guide_topk: int = 32
    """Number of atoms to keep per guide vector when sparsity_type is 'topk'. Ignored when sparsity_type is 'l1'."""

@dataclass
class SchedulerCfg:
    """Scheduler configuration."""

    type: str
    """torch.optim.lr_scheduler class name."""
    params: Dict[str, Any] = field(default_factory=dict)
    """Parameters passed to the scheduler constructor."""


@dataclass
class EmaCfg:
    """Exponential moving average configuration."""

    enabled: bool = False
    """Enable EMA for model parameters."""
    decay: float = 0.999
    """EMA decay factor."""
    use_buffers: bool = True
    """Whether to include buffers in EMA."""
    warmup_steps: int = 0
    """Skip EMA updates until this global step."""


@dataclass
class MethodConfig:
    """Method-specific configuration."""

    guide_sae: Optional[GuideSAECfg] = None
    """Guide SAE configuration."""


@dataclass
class OptionalModulesCfg:
    """Optional module configuration."""

    weight_module: Optional[WeightModuleCfg] = None
    """Per-layer weighting module configuration."""


class PipelineMethod(Enum):
    GUIDE_SAE = "guide_sae"


@dataclass
class ModelCfg:
    id: str
    """Model identifier (may include slashes)."""
    id_wo_slash: str
    """Identifier with slashes replaced."""
    encoder: str
    """Backbone encoder name."""
    image_size: int
    """Input image resolution."""
    features_list: List[int]
    """Feature layer indices to extract."""
    method: PipelineMethod
    """Pipeline method to use."""
    method_config: MethodConfig
    """Config for the selected method."""
    optional_modules: Optional[OptionalModulesCfg] = None
    """Optional auxiliary modules."""
    sigma: int = 4
    """Gaussian smoothing sigma for maps."""
    temperature: float = 1.0
    """Softmax temperature."""


@dataclass
class DataCfg:
    dataset_name: str
    """Dataset identifier."""
    input_type: str
    """'image' or 'feature' input mode."""
    path: str
    """Path to dataset split/config."""
    root: str
    """Root directory of datasets."""
    filter_kw: Optional[Dict[str, Any]] = None
    """Filtering keywords/options."""
    combined_datasets: Optional[List[Dict[str, Any]]] = None
    """Optional list of datasets to combine."""
    shuffle: bool = True
    """Shuffle data loader."""


@dataclass
class TrainCfg:
    epoch: int
    """Number of training epochs."""
    learning_rate: float
    """Base learning rate."""
    batch_size: int
    """Training batch size."""
    scheduler: Optional[SchedulerCfg] = None
    """Optional learning rate scheduler configuration."""
    ema: Optional[EmaCfg] = None
    """Optional EMA configuration."""


@dataclass
class MetricsCfg:
    image_level: Optional[List[str]] = None
    """Image-level metrics to compute."""
    pixel_level: Optional[List[str]] = None
    """Pixel-level metrics to compute."""


@dataclass
class ImageScoreCfg:
    """How to compute image-level score at eval: pipeline only, map-only, or hybrid."""

    mode: Literal["pipeline", "map", "hybrid"] = "pipeline"
    """pipeline=use pipeline anomaly_score as-is; map=aggregate from anomaly_map; hybrid=weighted sum."""

    map_pool: Literal["max", "top_q_mean", "log_sum_exp"] = "max"
    """When using map: max, mean of top q%% pixels, or temperature-scaled log-sum-exp."""

    top_q_percent: float = 10.0
    """Percent of pixels (0-100) for top_q_mean map_pool."""

    map_pool_tau: float = 1.0
    """Temperature for log_sum_exp: tau->0 => max, tau->infty => mean. Only used when map_pool=log_sum_exp."""

    alpha: float = 0.5
    """Weight for pipeline score in hybrid: alpha * pipeline + (1 - alpha) * map."""


@dataclass
class EvaluateCfg:
    batch_size: int
    """Evaluation batch size."""
    ckpt: str
    """Checkpoint path to load."""
    epoch: int
    """Target epoch number."""
    use_ema: bool = False
    """Use EMA model weights for evaluation."""
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    """Metrics configuration."""
    by_specie: bool = False
    """Enable species-level evaluation."""
    save_maps: bool = False
    """Persist anomaly maps to disk."""
    map_dtype: str = "float32"
    """Dtype for saved maps (float32 or float16)."""
    maps_dir: Optional[str] = None
    """Output directory for anomaly maps (defaults to base_dir/artifacts/...)."""

    image_score: Optional[ImageScoreCfg] = None
    """Image-level score source at eval: pipeline / map / hybrid. None = pipeline only."""

    pro_use_fast: bool = False
    """Use fast PRO implementation (bincount-based). If False, use original regionprops-based. Toggle to True after verifying same results."""


@dataclass
class CacheCfg:
    dir: str
    """Directory to store caches."""


@dataclass
class VisualizeCfg:
    enabled: bool = False
    """Enable visualization output."""
    metric: str = "f1"
    """Metric to drive visualization selection."""
    alpha: float = 0.5
    """Overlay alpha."""
    boundary_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Boundary color for overlays."""
    boundary_mode: str = "thick"
    """Boundary rendering style."""
    save_inputs: bool = False
    """Save input images."""
    save_anomaly: bool = False
    """Save anomaly maps."""
    save_overlay: bool = True
    """Save overlay images."""


__all__ = [
    "WeightModuleCfg",
    "GuideSAECfg",
    "SchedulerCfg",
    "EmaCfg",
    "MethodConfig",
    "OptionalModulesCfg",
    "PipelineMethod",
    "ModelCfg",
    "DataCfg",
    "TrainCfg",
    "MetricsCfg",
    "ImageScoreCfg",
    "EvaluateCfg",
    "CacheCfg",
    "VisualizeCfg",
]
