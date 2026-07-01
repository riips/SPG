"""Base-level configuration (aggregating common + SAE config)."""

from dataclasses import dataclass, field
from typing import List, Optional

from .types_common import (
    ModelCfg,
    DataCfg,
    TrainCfg,
    EvaluateCfg,
    CacheCfg,
    VisualizeCfg,
)
from .types_sae import SAECfg


@dataclass
class BaseCfg:
    mode: str
    """Run mode (e.g., train/eval)."""
    base_dir: str
    """Project base directory."""
    save_dir: str
    """Output save directory."""
    model: ModelCfg
    """Model configuration."""
    data: DataCfg
    """Training data configuration."""
    test_data: DataCfg
    """Test data configuration."""
    train: TrainCfg
    """Training hyperparameters."""
    evaluate: EvaluateCfg
    """Evaluation settings."""
    cache: CacheCfg
    """Cache settings."""
    seed: int = 42
    """Random seed."""
    print_freq: int = 1
    """Logging frequency (epochs)."""
    save_freq: int = 1
    """Checkpoint save frequency (epochs)."""

    # for eval
    train_dir: Optional[str] = None
    """Path to training outputs for eval."""
    visualize: Optional[VisualizeCfg] = None
    """Visualization options."""

    # for train SAE
    sae: Optional[List[SAECfg]] = None
    """SAE configurations."""


__all__ = ["BaseCfg"]

