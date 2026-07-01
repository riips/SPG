"""Utilities for building SAE token tensors `(B,T,D)` from batch inputs."""
from __future__ import annotations
from typing import Any, Callable, Optional, Sequence, Tuple, List

import numpy as np
import torch


@torch.no_grad()
def build_tokens_from_batch(
    batch: Any,
    device: torch.device | str,
    *,
    encoder: Any | None = None,
    features_list: Any | None = None,
) -> tuple[torch.Tensor, list[str] | None]:
    """Build SAE input `x: (B,T,D)` from a batch.

    - feature mode: use `batch.feats (B,D)` and `batch.mid_feats[-1] (B,N+1,D)`
    - image mode: call `encoder.encode_image(image, features_list)`

    Returns:
        x: `(B,T,D)` where T = 1 + number of patches, with `token==0` assumed to be CLS.
        cls_names: `list[str] | None`
    Example:
        x, cls = build_tokens_from_batch(batch, device="cuda", encoder=enc, features_list=["..."])
    """
    device = torch.device(device)

    # 1) Batch produced by DataManager collate (preferred path).
    if (
        hasattr(batch, "feats")
        and isinstance(batch.feats, torch.Tensor)
        and batch.feats.dim() == 2  # (B, D)
        and isinstance(getattr(batch, "mid_feats", None), list)
        and len(batch.mid_feats) > 0
        and isinstance(batch.mid_feats[-1], torch.Tensor)
        and batch.mid_feats[-1].dim() == 3  # (B, N+1, D)
    ):
        image_features = batch.feats.to(device)
        patches = batch.mid_feats[-1].to(device)
        x = torch.cat([image_features.view(image_features.shape[0], 1, -1),  # (B, 1, D)
                       patches[:, 1:]], dim=1)                                # (B, N, D)
        cls_names: list[str] | None = None
        if hasattr(batch, "cls_name"):
            if isinstance(batch.cls_name, list):
                cls_names = [str(c) for c in batch.cls_name]
            elif isinstance(batch.cls_name, torch.Tensor):
                cls_names = None
            elif isinstance(batch.cls_name, str):
                cls_names = [batch.cls_name] * x.shape[0]
        return x, cls_names

    # 2) Default collate (list[ImageRecord]) or a single ImageRecord.
    def _from_record(rec: Any) -> tuple[torch.Tensor, str | None]:
        if getattr(rec, "feats", None) is not None and isinstance(getattr(rec, "mid_feats", None), list):
            img_f = rec.feats.to(device).view(1, -1)                # (1, D)
            patches = rec.mid_feats[-1].to(device).unsqueeze(0)     # (1, N+1, D)
        else:
            if encoder is None:
                raise ValueError("imageモードのバッチは encoder が必要です（encode_image を利用）")
            if isinstance(getattr(rec, "img", None), torch.Tensor):
                image = rec.img.to(device).unsqueeze(0)             # (1, C, H, W)
            else:
                image = rec.img
            img_f, mid_list = encoder.encode_image(image=image, features_list=features_list)  # type: ignore
            patches = mid_list[-1]                                   # (1, N+1, D)
        x = torch.cat([img_f.view(1, 1, -1), patches[:, 1:]], dim=1) # (1, T, D)
        return x, getattr(rec, "cls_name", None)

    if isinstance(batch, list) and len(batch) > 0 and hasattr(batch[0], "img_path"):
        xs: list[torch.Tensor] = []
        names: list[str | None] = []
        for rec in batch:
            _x, _name = _from_record(rec)
            xs.append(_x)
            names.append(_name)
        x = torch.cat(xs, dim=0)
        cls_names = [n if n is not None else "unknown" for n in names] if any(n is not None for n in names) else None
        return x, cls_names

    if hasattr(batch, "img_path"):
        x, name = _from_record(batch)
        return x, [name] if name is not None else None

    raise ValueError("未知のバッチ形式です。DataLoader（featureモード）か list[ImageRecord] を渡してください。")

