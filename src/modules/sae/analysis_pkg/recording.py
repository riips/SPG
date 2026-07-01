"""Recording and online aggregation for activation events.

Note: Persisting events continuously can be expensive on large datasets.
By default, `metrics.activation_stats` uses a lightweight path that does not
save events or metadata. Explicitly pass `save_events=True` or a sink such as
Parquet only when needed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .tokens import build_tokens_from_batch
from .sinks import EventSink


@dataclass
class Accumulator:
    """Accumulator that stores per-atom statistics."""
    sum_all: torch.Tensor
    sum_active: torch.Tensor
    cnt_active: torch.Tensor
    max: torch.Tensor
    cnt_tokens: int
    topk_count: torch.Tensor | None


@dataclass
class RecordingSession:
    """State container for a recording session."""
    events: Dict[str, list]
    sample_meta: Dict[int, Dict[str, Any]]
    next_sample_id: int
    T: Optional[int]
    C: int
    opts: Dict[str, Any]
    acc_global: Accumulator
    acc_by_cls: Dict[str, Accumulator]


def new_accumulator(C: int, topk: int | None) -> Accumulator:
    """Initialize and return an accumulator."""
    return Accumulator(
        sum_all=torch.zeros(C, dtype=torch.float64),
        sum_active=torch.zeros(C, dtype=torch.float64),
        cnt_active=torch.zeros(C, dtype=torch.float64),
        max=torch.full((C,), float("-inf")),
        cnt_tokens=0,
        topk_count=torch.zeros(C, dtype=torch.float64) if topk is not None else None,
    )


def start_recording(
    *,
    C: int,
    threshold: float = 0.0,
    topk: int | None = 8,
    per_class: bool = False,
    save_events: bool = True,
    save_meta: bool = True,
) -> RecordingSession:
    """Create a new recording session.

    Args:
        C: Number of atoms
        threshold: Record events where z > threshold
        topk: Also record the top-k entries for each token; None disables this
        per_class: Keep per-class aggregation
        save_events: Whether to keep sparse event arrays in the session
        save_meta: Whether to keep sample_meta
    """
    return RecordingSession(
        events={"sample": [], "token": [], "atom": [], "value": []} if save_events else {"sample": [], "token": [], "atom": [], "value": []},
        sample_meta={} if save_meta else {},
        next_sample_id=0,
        T=None,
        C=C,
        opts={"threshold": threshold, "topk": topk, "per_class": per_class, "save_events": save_events, "save_meta": save_meta},
        acc_global=new_accumulator(C, topk),
        acc_by_cls={},
    )


def _accumulate(acc: Accumulator, z_cpu: torch.Tensor, active: torch.Tensor, topk: int | None):
    """Internal: add online aggregates to an accumulator."""
    C = z_cpu.shape[-1]
    acc.sum_all += z_cpu.sum(dim=(0, 1)).to(torch.float64)
    acc.sum_active += (z_cpu * active).sum(dim=(0, 1)).to(torch.float64)
    acc.cnt_active += active.sum(dim=(0, 1)).to(torch.float64)
    acc.max = torch.maximum(acc.max, z_cpu.amax(dim=(0, 1)))
    acc.cnt_tokens += int(z_cpu.shape[0] * z_cpu.shape[1])
    if topk is not None:
        tk = torch.topk(z_cpu, k=min(topk, C), dim=-1).indices
        binc = torch.bincount(tk.reshape(-1).to(torch.int64), minlength=C).to(torch.float64)
        assert acc.topk_count is not None
        acc.topk_count += binc


@torch.no_grad()
def record_batch(
    sess: RecordingSession,
    batch: Any,
    *,
    sae: Any,
    device: torch.device | str,
    encoder: Any | None = None,
    features_list: Any | None = None,
    class_extractor: Any | None = None,
    sink: EventSink | None = None,
) -> None:
    """Encode a batch and accumulate sparse events based on threshold/top-k settings."""
    topk = sess.opts["topk"]
    threshold = sess.opts["threshold"]
    want_cls = sess.opts["per_class"]
    save_events = bool(sess.opts.get("save_events", True))
    save_meta = bool(sess.opts.get("save_meta", True))

    x, cls_names = build_tokens_from_batch(batch, device=device, encoder=encoder, features_list=features_list)  # (B,T,D)
    z = sae.encode(x)  # (B,T,C)
    B, T, C = z.shape
    if sess.T is None:
        sess.T = int(T)

    z_cpu = z.detach().to("cpu")
    active = z_cpu > threshold

    # Online aggregation (global).
    _accumulate(sess.acc_global, z_cpu, active, topk)

    # Per-class aggregation.
    if want_cls:
        if cls_names is None and class_extractor is not None:
            try:
                cls_names = class_extractor(batch)
            except Exception:
                cls_names = None
        if cls_names is not None and len(cls_names) == B:
            for i, cname in enumerate(cls_names):
                if cname not in sess.acc_by_cls:
                    sess.acc_by_cls[cname] = new_accumulator(C, topk)
                _accumulate(sess.acc_by_cls[cname], z_cpu[i:i+1], active[i:i+1], topk)

    # Assign sample IDs and register metadata.
    sample_base = sess.next_sample_id
    has_gid = hasattr(batch, "global_id")
    has_path = hasattr(batch, "img_path")
    has_anom = hasattr(batch, "anomaly")
    has_spec = hasattr(batch, "specie_name")
    for b in range(B):
        sid = sample_base + b
        if save_meta:
            meta = sess.sample_meta.get(sid, {})
            meta["cls"] = cls_names[b] if (cls_names is not None and b < len(cls_names)) else meta.get("cls", None)
            meta["ds_index"] = int(sid)
            if has_gid:
                gid_val = batch.global_id[b]
                if isinstance(gid_val, torch.Tensor):
                    gid_val = int(gid_val.item())
                else:
                    gid_val = int(gid_val)
                meta["global_id"] = gid_val
            if has_path:
                meta["img_path"] = batch.img_path[b]
            if has_anom:
                an_val = batch.anomaly[b]
                if isinstance(an_val, torch.Tensor):
                    an_val = int(an_val.item())
                else:
                    an_val = int(an_val)
                meta["anomaly"] = an_val
            if has_spec:
                sp = batch.specie_name[b]
                if isinstance(sp, torch.Tensor):
                    sp = str(sp.item())
                meta["specie_name"] = str(sp)
            sess.sample_meta[sid] = meta
    sess.next_sample_id += B

    # Threshold events.
    if threshold is not None and threshold > -float("inf"):
        b_idx, t_idx, a_idx = torch.where(active)
        samp_np = (b_idx + sample_base).numpy()
        tok_np = t_idx.numpy()
        atm_np = a_idx.numpy()
        val_np = z_cpu[b_idx, t_idx, a_idx].numpy()
        if save_events:
            sess.events["sample"].append(samp_np)
            sess.events["token"].append(tok_np)
            sess.events["atom"].append(atm_np)
            sess.events["value"].append(val_np)
        if sink is not None:
            sink.on_events(samp_np, tok_np, atm_np, val_np, sess.sample_meta if save_meta else None)

    # Top-k events.
    if topk is not None:
        tk_idx = torch.topk(z_cpu, k=min(topk, C), dim=-1).indices  # (B,T,k)
        B_, T_, K_ = tk_idx.shape
        b_rep = torch.arange(B_).view(B_, 1, 1).repeat(1, T_, K_)
        t_rep = torch.arange(T_).view(1, T_, 1).repeat(B_, 1, K_)
        samp_np2 = (b_rep + sample_base).reshape(-1).numpy()
        tok_np2 = t_rep.reshape(-1).numpy()
        atm_np2 = tk_idx.reshape(-1).numpy()
        val_np2 = torch.gather(z_cpu, -1, tk_idx).reshape(-1).numpy()
        if save_events:
            sess.events["sample"].append(samp_np2)
            sess.events["token"].append(tok_np2)
            sess.events["atom"].append(atm_np2)
            sess.events["value"].append(val_np2)
        if sink is not None:
            sink.on_events(samp_np2, tok_np2, atm_np2, val_np2, sess.sample_meta if save_meta else None)


def _finalize_acc(acc: Accumulator) -> Dict[str, Any]:
    """Internal: finalize an accumulator and format it as a dictionary."""
    sum_all = acc.sum_all
    sum_active = acc.sum_active
    cnt_active = acc.cnt_active
    max_v = acc.max
    cnt_tokens = acc.cnt_tokens
    tk = acc.topk_count
    mean_all = (sum_all / max(cnt_tokens, 1)).numpy()
    mean_active = torch.where(cnt_active > 0, sum_active / cnt_active, torch.zeros_like(sum_active)).numpy()
    usage_rate = (cnt_active / max(cnt_tokens, 1)).numpy()
    out: Dict[str, Any] = {
        "usage_rate": usage_rate,
        "mean_active": mean_active,
        "mean_all": mean_all,
        "max": max_v.numpy(),
        "count_tokens": int(cnt_tokens),
        "count_active": cnt_active.numpy(),
    }
    if tk is not None:
        out["topk_rate"] = (tk / max(cnt_tokens, 1)).numpy()
    return out


def finalize_recording(sess: RecordingSession) -> Dict[str, Any]:
    """Finalize a recording session and return sparse events, statistics, and metadata."""
    # Concatenate event arrays.
    events: Dict[str, np.ndarray] = {}
    for k in ("sample", "token", "atom", "value"):
        if len(sess.events[k]) > 0:
            events[k] = np.concatenate(sess.events[k], axis=0)
        else:
            events[k] = np.empty((0,), dtype=np.int64 if k != "value" else np.float32)

    stats = {"global": _finalize_acc(sess.acc_global)}
    if len(sess.acc_by_cls) > 0:
        stats["by_class"] = {k: _finalize_acc(v) for k, v in sess.acc_by_cls.items()}

    return {
        "events": events,
        "stats": stats,
        "T": sess.T,
        "C": sess.C,
        "sample_meta": sess.sample_meta,
    }

