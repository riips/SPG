import logging
from typing import List, Tuple

from PIL import Image
import torch
import torchvision  # type: ignore
from transformers.models.siglip import (
    SiglipImageProcessor,
    SiglipProcessor,
    SiglipTokenizer,
    SiglipVisionModel,
)

from configs import ModelCfg

from .base_encoder import BaseEncoder

log = logging.getLogger(__name__)
torchvision.disable_beta_transforms_warning()


class SigLIP(BaseEncoder):
    def __init__(
        self, encoder_name: str, device: str | torch.device, model_cfg: ModelCfg
    ) -> None:
        super().__init__(encoder_name, device)
        self.cfg = model_cfg
        self._load_model()

    def _load_model(self) -> None:
        """Loads backbone weights and initialises the processor."""
        self.model = SiglipVisionModel.from_pretrained(self.cfg.id)  # type: ignore
        self.processor = self._get_processor(image_size=self.cfg.image_size)
        self.output_dim = self.model.config.hidden_size

    def _get_processor(self, image_size: int) -> SiglipProcessor:
        """Creates a SiglipProcessor with custom resize.

        Work-around:
            *`siglip2` checkpoints lack a compatible tokenizer; we therefore
            fall back to the original SigLIP tokenizer.*  See comment inline.

        Args:
            image_size: Target square resize (H = W).

        Returns:
            SiglipProcessor: Processor with ``processor.size`` overridden.
        """
        image_processor = SiglipImageProcessor.from_pretrained(  # type: ignore
            self.cfg.id,
            do_convert_rgb=True,
            size={"height": image_size, "width": image_size},
        )

        # tokenizer fallback to avoid siglip2 → sentencepiece mismatch
        tk_name = (
            "google/siglip-so400m-patch14-384"
            if "siglip2" in self.cfg.id
            else self.cfg.id
        )
        tokenizer = SiglipTokenizer.from_pretrained(tk_name)  # type: ignore

        processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)  # type: ignore

        return processor

    def preprocess(self, images):  # type: ignore
        """Preprocesses a single image or list of images for SigLIP."""
        return self.processor(images)  # type: ignore

    def encode_image(
        self,
        image: Image.Image | torch.Tensor,
        *,
        features_list: List[int] | Tuple[int, ...] = []
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns global + selected patch features from the backbone.

        Args:
            image: PIL image or Tensor in (C, H, W) format.
            features_list: Indices of hidden layers to extract as patch maps.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:

                * **image_features** – Tensor ``(B, D)`` pooled embedding.
                * **patch_features_list** – Tuple of tensors, each
                  ``(B, N_patches+1, D)`` (CLS + patch tokens).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        inputs["output_hidden_states"] = True
        inputs["interpolate_pos_encoding"] = True

        outputs = self.model(**inputs)

        # ------ global embedding (pooler) ---------------------------------
        image_features = outputs.pooler_output
        image_features = torch.stack(list(image_features))  # (B, D)

        # ------ patch features -------------------------------------------
        # SigLIP has no CLS token; prepend zero dummy so output is (B, 1+N, D) for pipeline
        patch_features_list = outputs.hidden_states  # tuple(len=ViT_depth, (B, N, D))
        p_list: List[torch.Tensor] = []
        for idx, patch_features in enumerate(patch_features_list):
            if idx in features_list:
                patch_features = self.model.vision_model.post_layernorm(patch_features)
                B, N, D = patch_features.shape
                cls_dummy = torch.zeros(B, 1, D, device=patch_features.device, dtype=patch_features.dtype)
                patch_features = torch.cat([cls_dummy, patch_features], dim=1)
                p_list.append(patch_features)
        patch_features_list = p_list

        return image_features, patch_features_list
