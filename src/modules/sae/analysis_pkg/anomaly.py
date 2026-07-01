"""Simple anomaly-detection utilities that treat dictionary atoms as anomaly prototypes."""
from __future__ import annotations
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from .tokens import build_tokens_from_batch


@torch.no_grad()
def anomaly_with_atoms(
    sae: Any,
    data: Iterable[Any],
    anomaly_atoms: Sequence[int],
    *,
    device: torch.device | str,
    encoder: Any | None = None,
    features_list: Any | None = None,
    agg: str = "max",
    temperature: float = 1.0,
    per_token_map: bool = False,
    grid_size: Tuple[int, int] | None = None,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """Compute anomaly scores from prototype similarity for CLS/tokens."""
    sae.eval()
    device = torch.device(device)

    Dmat = sae.decoder.dictionary.matrix.detach().to(device)  # (C, D)
    idx = torch.as_tensor(list(anomaly_atoms), dtype=torch.long, device=device)
    proto = F.normalize(Dmat.index_select(0, idx), dim=-1)  # (K, D)

    scores: list[float] = []
    tops: list[int] = []
    maps: list[np.ndarray] = []

    for b_idx, batch in enumerate(data):
        if max_batches is not None and b_idx >= max_batches:
            break

        x, _ = build_tokens_from_batch(batch, device=device, encoder=encoder, features_list=features_list)
        B, T, D = x.shape

        x_cls = F.normalize(x[:, 0, :], dim=-1)  # (B, D)
        sim_cls = x_cls @ proto.T

        if agg == "max":
            s, top = sim_cls.max(dim=1)
        elif agg == "mean":
            s = sim_cls.mean(dim=1)
            top = sim_cls.argmax(dim=1)
        elif agg == "lse":
            t = max(float(temperature), 1e-6)
            s = (sim_cls / t).logsumexp(dim=1) * t
            top = sim_cls.argmax(dim=1)
        else:
            raise ValueError("agg must be one of {'max','mean','lse'}")

        scores.extend(s.detach().to("cpu").tolist())
        tops.extend(idx[top].detach().to("cpu").tolist())

        if per_token_map:
            x_tok = F.normalize(x[:, 1:, :], dim=-1)  # (B, T-1, D)
            sim_tok = torch.einsum("btd,kd->btk", x_tok, proto)
            sim_tok_max = sim_tok.max(dim=-1).values  # (B, T-1)

            if grid_size is None:
                g = int((T - 1) ** 0.5)
                assert g * g == (T - 1), "grid_size を指定してください。"
                H = W = g
            else:
                H, W = grid_size
                assert H * W == (T - 1), "grid_size とトークン数が一致しません。"

            for b in range(B):
                maps.append(sim_tok_max[b].detach().to("cpu").view(H, W).numpy())

    out: Dict[str, Any] = {
        "score": np.asarray(scores, dtype=np.float32),
        "top_atom": np.asarray(tops, dtype=np.int64),
    }
    if per_token_map:
        out["maps"] = maps
    return out


def top_images_for_atom(
    rec_out: Dict[str, Any],
    atom_idx: int,
    *,
    topn: int = 20,
    reduce: str = "max",
) -> list[Dict[str, Any]]:
    """Return the top samples with strong responses for a given atom from recordings."""
    ev = rec_out["events"]
    sid = ev["sample"]; aid = ev["atom"]; val = ev["value"]
    mask = (aid == atom_idx)
    if mask.sum() == 0:
        return []
    sid_sel = sid[mask]; val_sel = val[mask]

    uniq, inv = np.unique(sid_sel, return_inverse=True)
    if reduce == "max":
        agg = np.full(uniq.shape, -np.inf, dtype=np.float32)
        np.maximum.at(agg, inv, val_sel)
    elif reduce == "sum":
        agg = np.zeros_like(uniq, dtype=np.float32)
        np.add.at(agg, inv, val_sel)
    else:
        raise ValueError("reduce must be 'max' or 'sum'")

    order = np.argsort(-agg)[:topn]
    result = []
    meta = rec_out["sample_meta"]
    for idx in order:
        s = int(uniq[idx])
        result.append({
            "sample_id": s,
            "score": float(agg[idx]),
            "cls": meta.get(s, {}).get("cls", None),
        })
    return result

