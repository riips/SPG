from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Runs a forward pass.

        Args:
            *args: Positional inputs (depends on subclass).
            **kwargs: Keyword inputs (depends on subclass).

        Returns:
            Any: Output suitable for loss or metrics.
        """
        raise NotImplementedError  # pragma: no cover
