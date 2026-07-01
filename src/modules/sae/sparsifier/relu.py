import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSparsifier


class ReLUSparsifier(BaseSparsifier):
    """Standard ReLU sparsifier: max(pre, 0)."""

    def forward(self, pre: torch.Tensor) -> torch.Tensor:
        return F.relu(pre)


