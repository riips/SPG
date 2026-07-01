import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSparsifier


class JumpReLUSparsifier(BaseSparsifier):
    """
    JumpReLU: ReLU(pre - threshold)
    Supports both fixed and learnable thresholds.
    """

    def __init__(self, threshold: float = 0.1, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        else:
            self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, pre: torch.Tensor) -> torch.Tensor:
        return F.relu(pre - self.threshold)

