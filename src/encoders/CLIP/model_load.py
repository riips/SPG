import hashlib
import os
import urllib.request
import warnings
from typing import List, Union

import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from .build_model import build_model

__all__ = ["available_models", "load"]

_MODELS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, cache_dir: Union[str, None] = None) -> str:
    cache_dir = os.path.expanduser(cache_dir or "~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if not expected_sha256:
            return download_target
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(
            expected_sha256
        ):
            return download_target
        warnings.warn(
            f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
        )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.headers.get("Content-Length", 0)),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(
        open(download_target, "rb").read()
    ).hexdigest().startswith(expected_sha256):
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def available_models() -> List[str]:
    return list(_MODELS.keys())


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str | None = None,
):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, "rb") as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(
                    f"File {model_path} is not a JIT archive. Loading as a state dict instead"
                )
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    raise NotImplementedError("JIT CLIP loading is not supported by this OpenCLIP loader")
