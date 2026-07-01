from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class SAECfg:
    """Configuration for Sparse Autoencoder."""

    hidden_dim: int
    """Hidden dimension (=number of atoms, dictionary size)."""

    sparsifier_kind: str
    """Type of sparsifier function."""

    sparsifier_params: Dict[str, Any]
    """Parameters for the sparsifier function."""

    sparsity_penalty_type: str
    """Type of sparsity penalty."""

    lmbda: float
    """Coefficient for sparsity penalty."""

    recon_error_type: str
    """Type of reconstruction error."""

    clamp_dict_nonneg: bool
    """Whether to clamp dictionary atoms to be non-negative."""

    input_norm: str = "none"
    """Input normalization before SAE encode: 'none' or 'l2'."""

    use_cls: bool = True
    """Whether to include the CLS token in SAE inputs (True) or drop it and use only patch tokens (False)."""

    name: Optional[str] = "sae"
    """Identifier name of the SAE."""

    hidden_dim_anomaly: Optional[int] = None
    """Hidden dimension for anomaly SAE."""

    encoder_bias_enabled: bool = False
    """Whether to enable bias in the encoder."""

    dead_atom_rule: str = "count_zero"
    """Dead-atom detection rule: 'count_zero' or 'usage_rate_eps'."""

    dead_atom_eps: float = 1e-6
    """Threshold used by the 'usage_rate_eps' rule."""

    dead_atom_threshold: float = 0.0
    """Activation threshold (z > threshold)."""

    auxk: Optional[int] = None
    """AuxK k value. None disables AuxK."""

    auxk_coef: float = 1 / 32
    """Coefficient for the AuxK loss (alpha)."""

    dead_steps_threshold: int = 10_000_000
    """Step threshold for dead-atom detection, measured in batches."""

    dead_activation_threshold: float = 1e-3
    """Activation threshold for dead-atom detection (|pre| > threshold)."""
