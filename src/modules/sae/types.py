from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any
import torch

class SparsifierKind(Enum):
    RELU = "relu"
    TOPK = "topk"
    JUMP_RELU = "jump_relu"

class ReconErrorType(Enum):
    MSE = "mse"
    COS = "cos"

class SparsityPenaltyType(Enum):
    L1 = "l1"
    L0_PROXY = "l0_proxy"

@dataclass
class BatchOutput:
    """Output of SAE.

    Args:
        x: Input tensor.
        z: Encoded tensor.
        x_hat: Reconstructed tensor.
        recon_error: Reconstruction error.
        sparsity_penalty: Sparsity penalty.
        aux: Auxiliary output.
    """
    x: torch.Tensor
    """Input tensor."""
    z: torch.Tensor
    """Encoded tensor."""
    x_hat: torch.Tensor
    """Reconstructed tensor."""
    recon_error: torch.Tensor
    """Reconstruction error."""
    sparsity_penalty: torch.Tensor
    """Sparsity penalty."""
    sparsify_z: bool
    """Whether to sparsify the encoded tensor."""
    aux: Dict[str, Any] # for future use
    """Auxiliary output for future use."""