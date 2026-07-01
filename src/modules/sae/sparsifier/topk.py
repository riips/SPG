import torch
import torch.nn as nn

from .base import BaseSparsifier


class TopKSparsifier(BaseSparsifier):
    """Keep top-k along feature-dim, zero out others; optionally clamp non-negative."""

    def __init__(self, topk: int = 32, clamp_nonneg: bool = True) -> None:
        super().__init__()
        self.topk = topk
        self.clamp_nonneg = clamp_nonneg

    def forward(self, pre: torch.Tensor) -> torch.Tensor:
        k = min(self.topk, pre.shape[-1])
        vals, idx = torch.topk(pre, k, dim=-1)
        if self.clamp_nonneg:
            vals = vals.clamp_min(0)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z


