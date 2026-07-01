"""encoder_manager.py

Factory-style helper that instantiates an image-text encoder
(OpenCLIP / SigLIP / DINO) given a model configuration.

The class keeps exactly one live model and exposes a simple API:

    >>> encoder_manager = EncoderManager(cfg.model)
    >>> backbone = encoder_manager.load_encoder()  # or encoder_manager.load_encoder("ViT-L/14@336px")
    >>> output = backbone.encode_image(img)
    >>> encoder_manager.clear_encoder()  # release for GC / VRAM
"""

from typing import Dict, Type

import torch

from configs import ModelCfg

from .base_encoder import BaseEncoder
from .dino import DINO
from .openclip import OpenCLIP
from .siglip import SigLIP

ENCODER_MAP: Dict[str, Type[BaseEncoder]] = {
    "OpenCLIP": OpenCLIP,
    "SigLIP": SigLIP,
    "DINO": DINO,
}


class EncoderManager:
    """Thin wrapper that lazily constructs a chosen backbone.

    Args:
        model_cfg: Hydra / OmegaConf model configuration.
        device: Optional torch device; if ``None`` defaults to
            ``"cuda"`` when available, otherwise ``"cpu"``.

    Attributes:
        model_cfg: Saved for later use when `load_encoder` is called.
        device: Target computation device.
        encoder: Cached encoder instance or ``None`` until loaded.
    """

    def __init__(self, model_cfg: ModelCfg, device: str | torch.device | None = None) -> None:
        self.model_cfg = model_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None

    def load_encoder(self, encoder_name: str | None = None) -> BaseEncoder:
        """Instantiates the backbone specified by *encoder_name* (or ``self.model_cfg.id``).

        Args:
            encoder_name: Optional override. If ``None``, uses ``self.model_cfg.id``.

        Returns:
            BaseEncoder: The created encoder placed on ``self.device``.

        Raises:
            KeyError: If ``self.model_cfg.encoder`` is not in ``ENCODER_MAP``.
        """
        encoder_name = encoder_name or self.model_cfg.id
        encoder_key = self.model_cfg.encoder
        if encoder_key not in ENCODER_MAP:
            raise KeyError(
                f"Unknown encoder '{encoder_key}'. Available: {list(ENCODER_MAP)}"
            )

        self.encoder = ENCODER_MAP[encoder_key](
            encoder_name, self.device, self.model_cfg
        )
        return self.encoder

    def clear_encoder(self) -> None:
        """Drops the cached encoder reference for manual VRAM cleanup."""
        self.encoder = None
