import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import BinaryDiceLoss, FocalLoss

def loss_function(
    name: str, logit: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Selects and applies a loss function by name.

    Supported keys: ``'focal'``, ``'dice'``, ``'cross_entropy'``,
    ``'binary_cross_entropy'``.  Defaults to cross-entropy.

    Args:
        name: Loss identifier.
        logit: Raw model logits (pre-activation).
        target: Target tensor (same shape / broadcastable).

    Returns:
        torch.Tensor: Scalar loss.
    """
    match name:
        case "focal":
            loss_func = FocalLoss()
        case "dice":
            loss_func = BinaryDiceLoss()
        case "cross_entropy":
            loss_func = F.cross_entropy
        case "binary_cross_entropy":
            loss_func = nn.BCEWithLogitsLoss()
        case "mse":
            loss_func = nn.MSELoss()
        case "mae":
            loss_func = nn.L1Loss()
        case _:  # default
            loss_func = F.cross_entropy
    loss: torch.Tensor = loss_func(logit, target)
    return loss

def compute_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors.

    Handles broadcasting so that either input can be 1-D (single vector)
    or 2-D (batch / patch matrix).

    Args:
        x1: Tensor ``(..., D)``.
        x2: Tensor ``(..., D)``.
        normalize: If ``True``, ℓ2-normalizes both inputs first.

    Returns:
        torch.Tensor: Cosine similarity with shape that follows broadcasting
        rules, e.g. ``(B1, B2)`` or ``(B, N_patches)``.

    Raises:
        ValueError: If the feature dimensions do not match or both tensors
        have rank > 2.
    """

    # Check if the last dimensions of the input tensors match
    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError(
            f"Last Dimension mismatch: {x1.shape[-1]} and {x2.shape[-1]} must be the same."
        )

    # Check if at least one of the tensors is 1D or 2D
    if x1.ndim not in [1, 2] and x2.ndim not in [1, 2]:
        raise ValueError(
            f"At least one of tensor_a or tensor_b must be a 1D or 2D tensor, "
            f"but got {x1.ndim}D and {x2.ndim}D tensors."
        )

    # Normalize the tensors
    if normalize:
        x1 = x1 / x1.norm(dim=-1, keepdim=True)  # type: ignore
        x2 = x2 / x2.norm(dim=-1, keepdim=True)  # type: ignore

    # Adjust the dimensions of the tensors
    if x1.ndim >= x2.ndim:  # type: ignore
        for _ in range(x1.ndim - x2.ndim):  # type: ignore
            x2 = x2.unsqueeze(0)  # type: ignore
    else:
        for _ in range(x2.ndim - x1.ndim):  # type: ignore
            x1 = x1.unsqueeze(0)  # type: ignore

    # Compute the cosine similarity
    return x1 @ x2.transpose(-2, -1)  # type: ignore

def softmax_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.00,
    dim: int = -1,
) -> torch.Tensor:
    return (logits / temperature).softmax(dim=dim)

def get_score_map_with_target_size(
    x: torch.Tensor, target_size: int
) -> torch.Tensor:
    """Reshapes patch scores into a square map and resizes to *target_size*.

    Args:
        x: Score tensor of shape ``(B, N_patches, C)`` where
            ``N_patches`` is assumed to be a perfect square.
        target_size: Desired output spatial size (H = W).

    Returns:
        torch.Tensor: Resized score map of shape ``(B, C, target_size, target_size)``.
    """

    # Reshape the score map
    side_length = int(x.shape[1] ** 0.5)
    x = x.reshape(
        x.shape[0], side_length, side_length, -1
    )  # [batch_size, side_length, side_length, channels]

    # Resize the score map
    x = x.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]
    x = torch.nn.functional.interpolate(x, size=(target_size, target_size), mode="bilinear")  # type: ignore

    return x  # type: ignore
