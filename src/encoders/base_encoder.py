from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn


# Encoder base class with a shared interface.
class BaseEncoder(nn.Module):
    def __init__(
        self, encoder_name: str, device: str | torch.device, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()  # type: ignore
        self.encoder_name = encoder_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        raise NotImplementedError

    def preprocess(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def encode_image(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

    def to_device(self, device: str | torch.device) -> None:
        self.device = device
        if self.model is not None:
            self.model.to(device)
