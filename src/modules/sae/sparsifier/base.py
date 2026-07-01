from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class BaseSparsifier(nn.Module, ABC):
    """Map pre-activation [B, T, C] -> activation [B, T, C]."""
    
    @abstractmethod
    def forward(self, pre: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError