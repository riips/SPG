from dataclasses import asdict
from typing import Any, Callable, Dict, List, Tuple

import torch
import torchvision  # type: ignore
from torchvision import transforms  # type: ignore

from .image_record import ImageRecord

torchvision.disable_beta_transforms_warning()


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

SIGLIP_MODELS = {
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-512",
    "google/siglip-large-patch16-384",
    "google/siglip-so400m-patch14-384",
    "google/siglip-so400m-patch14-224",
    "google/siglip-so400m-patch16-256-i18n",
    "google/siglip2-so400m-patch14-384",
    "google/siglip2-so400m-patch16-512",
    "google/siglip2-large-patch16-512",
}

DINO_MODELS = {
    "facebook/dino-vitb16",
    "facebook/dino-vitb8",
    "facebook/dino-vits16",
    "facebook/dino-vits8",
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}


def get_preprocess(model_name: str, image_size: int | None = None) -> Tuple[
    transforms.Compose | None,
    transforms.Compose,
    Callable[[List[ImageRecord]], ImageRecord],
]:
    match model_name:
        case "ViT-L/14@336px":
            if image_size is None:
                image_size = 336
            train_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.Lambda(lambd=lambda img: img.convert("RGB")),  # type: ignore
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD
                    ),
                ]
            )

            target_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )

            def collate_fn(batch: List[ImageRecord]) -> ImageRecord:
                sample_dict = asdict(batch[0])
                batch_dict: Dict[str, Any] = {}
                for _key in sample_dict.keys():
                    items = [getattr(b, _key) for b in batch]
                    match items[0]:
                        case int() | float():
                            items = torch.tensor(items)
                        case torch.Tensor():
                            items = torch.cat(items)
                        case list() as lst if lst and all(isinstance(x, torch.Tensor) for x in lst):  # type: ignore
                            items = [torch.cat(col, dim=0) for col in zip(*items)]
                        case _:
                            pass
                    batch_dict[_key] = items
                return ImageRecord(**batch_dict)

        case _model if _model in SIGLIP_MODELS or _model in DINO_MODELS:
            # SigLIP and DINO variants use the same preprocessing through Transformers.
            train_transform = None
            target_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )

            def collate_fn(batch: List[ImageRecord]) -> ImageRecord:
                sample_dict = asdict(batch[0])
                batch_dict: Dict[str, Any] = {}
                for _key in sample_dict.keys():
                    items = [getattr(b, _key) for b in batch]
                    match items[0]:
                        case int() | float():
                            items = torch.tensor(items)
                        case torch.Tensor():
                            items = torch.cat(items)
                        case list() as lst if lst and all(isinstance(x, torch.Tensor) for x in lst):  # type: ignore
                            items = [torch.cat(col, dim=0) for col in zip(*items)]
                        case _:
                            pass
                    batch_dict[_key] = items
                return ImageRecord(**batch_dict)

        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    return train_transform, target_transform, collate_fn
