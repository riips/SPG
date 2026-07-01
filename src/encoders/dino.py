from typing import Any, Dict, List, Tuple, cast

import torch
from transformers import AutoImageProcessor, AutoModel

from configs import ModelCfg

from .base_encoder import BaseEncoder


class DINO(BaseEncoder):
    def __init__(
        self, encoder_name: str, device: str | torch.device, model_cfg: ModelCfg
    ) -> None:
        super().__init__(encoder_name, device)
        self.cfg = model_cfg
        self._load_model()

    def _load_model(self) -> None:
        """Loads DINO/DINOv2/DINOv3 backbone weights and initialises the processor."""
        self.model = AutoModel.from_pretrained(self.cfg.id)  # type: ignore
        self.processor = AutoImageProcessor.from_pretrained(self.cfg.id)  # type: ignore
        self.output_dim = self.model.config.hidden_size  # type: ignore

        # Get the number of DINOv3 register tokens.
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)  # type: ignore
        self.patch_size = getattr(self.model.config, "patch_size", 16)  # type: ignore
        # Prefer model.image_size from YAML; otherwise use the model default.
        cfg_image_size = getattr(self.cfg, "image_size", None)
        self.image_size = cfg_image_size if cfg_image_size is not None else getattr(self.model.config, "image_size", 224)  # type: ignore

    def preprocess(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Preprocesses images for DINO."""
        return self.processor(  # type: ignore
            images,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )

    def encode_image(
        self, image: torch.Tensor, features_list: List[int] = [], **kwargs: Any
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns global + selected patch features from DINO backbone.

        Unified interface for DINOv1, DINOv2, and DINOv3.

        Args:
            image: PIL image or Tensor in (C, H, W) format.
            features_list: Indices of hidden layers to extract as patch maps.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                * **image_features** – Tensor ``(B, D)`` global embedding.
                * **patch_features_list** – List of tensors, each
                  ``(B, N_patches+1, D)`` (CLS + patch tokens).
        """
        inputs: Dict[str, torch.Tensor] = self.processor(  # type: ignore
            images=image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
            crop_size={"height": self.image_size, "width": self.image_size},
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # type: ignore
        inputs["output_hidden_states"] = True  # type: ignore

        outputs = self.model(**inputs)  # type: ignore

        # Global embedding (CLS token) for the unified interface.
        _last_hidden_state = outputs.last_hidden_state  # type: ignore
        image_features: torch.Tensor = cast(
            torch.Tensor, _last_hidden_state[:, 0]
        )  # (B, D)

        # Patch features from selected layers for the unified interface.
        # Register tokens are removed and CLS is prepended.
        patch_features_list: List[torch.Tensor] = []
        for idx, hidden_state in enumerate(outputs.hidden_states):  # type: ignore
            if idx in features_list:
                cls_token = hidden_state[:, :1, :]
                # Extract patch tokens dynamically.
                patch_features = self._extract_patch_tokens(hidden_state)  # type: ignore
                patch_features = torch.cat([cls_token, patch_features], dim=1)
                patch_features_list.append(patch_features)

        return image_features, patch_features_list

    def _extract_patch_tokens(self, hidden_state: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Extract patch tokens dynamically for DINOv1/v2/v3.

        Args:
            hidden_state: Hidden state tensor (B, N_tokens, D)

        Returns:
            torch.Tensor: Patch tokens (B, N_patches, D)
        """
        total_tokens = hidden_state.shape[1]

        if self.num_register_tokens > 0:
            # DINOv3: CLS(1) + Register(num_register) + Patches
            patch_start = 1 + self.num_register_tokens
            patch_features = hidden_state[:, patch_start:, :]

            expected_patches = (self.image_size // self.patch_size) ** 2
            if patch_features.shape[1] != expected_patches:
                # Fallback: infer the register-token count automatically.
                auto_register = total_tokens - 1 - expected_patches
                if 0 <= auto_register <= 16:
                    patch_start = 1 + auto_register
                    patch_features = hidden_state[:, patch_start:, :]
                else:
                    # Final check.
                    if patch_features.shape[1] != expected_patches:
                        raise RuntimeError(
                            f"Patch tokens size mismatch after fallback: got {patch_features.shape[1]}, expected {expected_patches}. "
                            f"(num_register_tokens={self.num_register_tokens}, total_tokens={total_tokens})"
                        )

        else:
            # DINOv1/v2: CLS(1) + Patches
            patch_features = hidden_state[:, 1:, :]

            expected_patches = (self.image_size // self.patch_size) ** 2
            if patch_features.shape[1] != expected_patches:
                # Fallback: remove leading non-patch tokens for cases with
                # special tokens other than CLS.
                auto_non_patch = total_tokens - expected_patches
                if 1 <= auto_non_patch <= 16:
                    patch_features = hidden_state[:, auto_non_patch:, :]
                # Final check.
                if patch_features.shape[1] != expected_patches:
                    raise RuntimeError(
                        f"Patch tokens size mismatch (v1/v2 path): got {patch_features.shape[1]}, expected {expected_patches}. "
                        f"(total_tokens={total_tokens})"
                    )

        return patch_features
