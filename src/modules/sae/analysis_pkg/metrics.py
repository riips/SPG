"""Metrics for reconstruction quality, activation statistics, CLS deviation analysis, and related tasks."""
from __future__ import annotations
from typing import Any, Dict, Iterable, Callable, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .tokens import build_tokens_from_batch
from .recording import start_recording, record_batch, finalize_recording


@torch.no_grad()
def reconstruction_quality(
    sae: Any,
    data: Iterable[Any],
    *,
    device: torch.device | str,
    encoder: Any | None = None,
    features_list: Any | None = None,
    max_batches: int | None = None,
    per_class: bool = False,
    class_extractor: Callable[[Any], list[str]] | None = None,
    cls_only: bool = False,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Evaluate reconstruction quality (R^2, MSE, mean cosine similarity) in streaming mode.

    - R^2 is sum-based: 1 - SSE_total / SST_total
    - cos is averaged per token

    Args:
        sae: SAE model
        data: Iterable such as a DataLoader
        device: Inference device
        encoder, features_list: Encoder settings for image mode
        max_batches: Early stopping limit
        per_class: Return per-class aggregation
        class_extractor: Class-name extraction function when collate cannot provide it
        cls_only: Evaluate only the CLS token
        eps: Epsilon to avoid a zero R^2 denominator
    Returns:
        dict: {"global": {...}, "by_class": {...?}}
    """
    def _new_acc() -> Dict[str, float | int]:
        return {
            "sse": 0.0,
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "elems": 0,
            "cos_sum": 0.0,
            "cos_cnt": 0,
        }

    total = _new_acc()
    by_cls: Dict[str, Dict[str, float | int]] = {}

    sae.eval()
    for b_idx, batch in enumerate(tqdm(data, leave=False, desc="reconstruction_quality")):
        if max_batches is not None and b_idx >= max_batches:
            break

        x, cls_names = build_tokens_from_batch(batch, device=device, encoder=encoder, features_list=features_list)
        x_hat, _ = sae.reconstruct(x)

        if cls_only:
            x_use = x[:, 0:1, :]
            xh_use = x_hat[:, 0:1, :]
        else:
            x_use = x
            xh_use = x_hat

        diff = (xh_use - x_use)
        sse = diff.pow(2).sum().item()
        sum_x = x_use.sum().item()
        sum_x2 = x_use.pow(2).sum().item()
        elems = int(x_use.numel())

        x_n = F.normalize(x_use, dim=-1)
        xh_n = F.normalize(xh_use, dim=-1)
        cos_bt = F.cosine_similarity(x_n, xh_n, dim=-1)  # (B,T)
        cos_sum = float(cos_bt.sum().item())
        cos_cnt = int(cos_bt.numel())

        total["sse"] += sse
        total["sum_x"] += sum_x
        total["sum_x2"] += sum_x2
        total["elems"] += elems
        total["cos_sum"] += cos_sum
        total["cos_cnt"] += cos_cnt

        if per_class:
            if cls_names is None and class_extractor is not None:
                try:
                    cls_names = class_extractor(batch)
                except Exception:
                    cls_names = None
            if cls_names is not None:
                B = x.shape[0]
                for i in range(B):
                    cname = str(cls_names[i])
                    if cname not in by_cls:
                        by_cls[cname] = _new_acc()
                    xi = x[i]; xhi = x_hat[i]
                    if cls_only:
                        xi = xi[0:1, :]
                        xhi = xhi[0:1, :]
                    di = (xhi - xi)
                    _sse = di.pow(2).sum().item()
                    _sum_x = xi.sum().item()
                    _sum_x2 = xi.pow(2).sum().item()
                    _elems = int(xi.numel())
                    _cos = F.cosine_similarity(
                        F.normalize(xi, dim=-1), F.normalize(xhi, dim=-1), dim=-1
                    )
                    _cos_sum = float(_cos.sum().item())
                    _cos_cnt = int(_cos.numel())
                    acc = by_cls[cname]
                    acc["sse"] += _sse
                    acc["sum_x"] += _sum_x
                    acc["sum_x2"] += _sum_x2
                    acc["elems"] += _elems
                    acc["cos_sum"] += _cos_sum
                    acc["cos_cnt"] += _cos_cnt

    def _finalize(acc: Dict[str, float | int]) -> Dict[str, float]:
        sse = float(acc["sse"])
        sum_x = float(acc["sum_x"])
        sum_x2 = float(acc["sum_x2"])
        elems = int(acc["elems"])
        sst = max(sum_x2 - (sum_x * sum_x) / max(elems, 1), 0.0)
        r2 = 1.0 - (sse / max(sst, eps)) if elems > 0 else float("nan")
        mse = sse / max(elems, 1) if elems > 0 else float("nan")
        cos = float(acc["cos_sum"]) / max(int(acc["cos_cnt"]), 1)
        return {"r2": r2, "mse": mse, "cos": cos, "sse": sse, "sst": sst, "elems": float(elems)}

    out: Dict[str, Any] = {"global": _finalize(total)}
    if per_class:
        out["by_class"] = {c: _finalize(a) for c, a in by_cls.items()}
    return out


@torch.no_grad()
def activation_stats(
    sae: Any,
    data: Iterable[Any],
    *,
    device: torch.device | str,
    encoder: Any | None = None,
    features_list: Any | None = None,
    threshold: float = 0.0,
    topk: int | None = None,
    max_batches: int | None = None,
    per_class: bool = False,
    class_extractor: Callable[[Any], list[str]] | None = None,
    save_events: bool = False,
    save_meta: bool = False,
) -> Dict[str, Any]:
    """Aggregate and return activation statistics for encoded z through the recording API.

    Returns:
        dict: {"global": {...}, "by_class": {...?}}
    """
    sae.eval()
    sess = start_recording(
        C=sae.hidden_dim,
        threshold=threshold,
        topk=topk,
        per_class=per_class,
        save_events=save_events,
        save_meta=save_meta,
    )
    for b_idx, batch in enumerate(
        tqdm(
            data,
            desc="activation_stats",
            total=len(data) if hasattr(data, "__len__") else None,
        )
    ):
        if max_batches is not None and b_idx >= max_batches:
            break
        record_batch(
            sess,
            batch,
            sae=sae,
            device=device,
            encoder=encoder,
            features_list=features_list,
            class_extractor=class_extractor,
            sink=None,
        )
    rec = finalize_recording(sess)
    return rec["stats"]


def analyze_cls_token_deviation(
    rec_out: Dict[str, Any],
    *,
    target_class: str | None = None,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Compute class-wise normal mean/std for CLS tokens and return per-atom differences
    and z-score differences for anomaly samples within the same class."""
    ev = rec_out["events"]
    meta: Dict[int, Dict[str, Any]] = rec_out.get("sample_meta", {})
    C: int = int(rec_out["C"])

    sid_arr = ev["sample"]
    tid_arr = ev["token"]
    aid_arr = ev["atom"]
    val_arr = ev["value"]

    cls_mask = (tid_arr == 0)
    sid_cls = sid_arr[cls_mask]
    aid_cls = aid_arr[cls_mask]
    val_cls = val_arr[cls_mask]

    all_classes = set(m.get("cls", None) for m in meta.values())
    classes = [target_class] if target_class is not None else sorted(list(all_classes))

    def build_z_cls_for_sid(sid: int):
        import numpy as _np
        z = _np.zeros((C,), dtype=_np.float32)
        m = (sid_cls == sid)
        if m.any():
            a_sel = aid_cls[m]
            v_sel = val_cls[m]
            _np.maximum.at(z, a_sel, v_sel)
        return z

    class_summaries: Dict[str | None, Dict[str, Any]] = {}
    abnormal_diffs: list[Dict[str, Any]] = []

    for cls_name in classes:
        normal_sids: list[int] = []
        abnormal_sids: list[int] = []

        for sid, m in meta.items():
            if m.get("cls", None) != cls_name:
                continue
            an = m.get("anomaly", None)
            if an == 0:
                normal_sids.append(int(sid))
            elif an == 1:
                abnormal_sids.append(int(sid))
            else:
                continue

        if len(normal_sids) == 0:
            continue

        import numpy as _np
        normals = _np.stack([build_z_cls_for_sid(s) for s in normal_sids], axis=0)
        mu = normals.mean(axis=0)
        sigma = normals.std(axis=0)

        class_summaries[cls_name] = {
            "mu": mu,
            "sigma": sigma,
            "n_normal": int(len(normal_sids)),
        }

        for sid in abnormal_sids:
            z = build_z_cls_for_sid(sid)
            diff = z - mu
            zscore = diff / (sigma + eps)
            pos_atoms = _np.flatnonzero(diff > 0).astype(_np.int64, copy=False)
            m = meta.get(sid, {})
            abnormal_diffs.append(
                {
                    "sample_id": int(sid),
                    "cls": cls_name,
                    "ds_index": int(m["ds_index"]) if "ds_index" in m else None,
                    "global_id": int(m["global_id"]) if "global_id" in m else None,
                    "diff": diff,
                    "zscore": zscore,
                    "pos_atoms": pos_atoms,
                }
            )

    return {
        "class_summaries": class_summaries,
        "abnormal_diffs": abnormal_diffs,
    }

