"""Utilities for inverted-index construction and heatmap/ranking recovery."""
from __future__ import annotations
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional

import numpy as np
import torch

from .tokens import build_tokens_from_batch


@torch.no_grad()
def build_atom_inverted_index(
    sae: Any,
    data: Iterable[Any],
    atom_ids: Sequence[int],
    *,
    device: torch.device | str,
    threshold: float = 0.0,
    encoder: Any | None = None,
    features_list: Any | None = None,
    exclude_cls_token: bool = True,
    max_batches: int | None = None,
) -> tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, Any]]]:
    """Build sparse (sample, token, value) events for selected atoms as an inverted index."""
    sae = sae.eval()
    device = torch.device(device)
    idx = torch.as_tensor(list(atom_ids), dtype=torch.long, device=device)

    index: Dict[int, Dict[str, list[np.ndarray]]] = {int(a): {"sample": [], "token": [], "value": []} for a in idx.tolist()}
    sample_meta: Dict[int, Dict[str, Any]] = {}

    next_sid = 0
    for b_idx, batch in enumerate(data):
        if max_batches is not None and b_idx >= max_batches:
            break

        x, _ = build_tokens_from_batch(batch, device=device, encoder=encoder, features_list=features_list)
        z = sae.encode(x)  # (B,T,C)
        B, T, C = z.shape

        zA = z[..., idx]
        mask = zA > threshold  # (B,T,K)
        if exclude_cls_token and T > 0:
            m = mask.clone()
            m[:, 0, :] = False
        else:
            m = mask

        if m.any():
            b_i, t_i, k_i = torch.where(m)
            v = zA[b_i, t_i, k_i]
            samp = (b_i + next_sid).to("cpu").numpy().astype(np.int32)
            tok = t_i.to("cpu").numpy().astype(np.int32)
            atm = idx[k_i].to("cpu").numpy().astype(np.int32)
            val = v.to("cpu").numpy().astype(np.float16)

            for a in np.unique(atm):
                sel = (atm == a)
                d = index[int(a)]
                d["sample"].append(samp[sel])
                d["token"].append(tok[sel])
                d["value"].append(val[sel])

        meta_keys = ("img_path", "cls_name", "anomaly", "global_id", "specie_name")
        has = {k: hasattr(batch, k) for k in meta_keys}
        for i in range(B):
            sid = next_sid + i
            m = sample_meta.get(sid, {})
            if has["img_path"]:
                m["img_path"] = batch.img_path[i] if isinstance(batch.img_path, list) else batch.img_path
            if has["cls_name"]:
                c = batch.cls_name[i] if isinstance(batch.cls_name, list) else batch.cls_name
                m["cls"] = str(c) if not isinstance(c, torch.Tensor) else None
            if has["anomaly"]:
                a = batch.anomaly[i]
                m["anomaly"] = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
            if has["global_id"]:
                g = batch.global_id[i]
                m["global_id"] = int(g.item()) if isinstance(g, torch.Tensor) else int(g)
            if has["specie_name"]:
                s = batch.specie_name[i] if isinstance(batch.specie_name, list) else batch.specie_name
                m["specie_name"] = str(s) if not isinstance(s, torch.Tensor) else str(s.item())
            sample_meta[sid] = m

        next_sid += B

    out_index: Dict[int, Dict[str, np.ndarray]] = {}
    for a, d in index.items():
        out_index[a] = {
            "sample": np.concatenate(d["sample"], axis=0) if d["sample"] else np.empty((0,), dtype=np.int32),
            "token":  np.concatenate(d["token"],  axis=0) if d["token"]  else np.empty((0,), dtype=np.int32),
            "value":  np.concatenate(d["value"],  axis=0) if d["value"]  else np.empty((0,), dtype=np.float16),
        }
    return out_index, sample_meta


@torch.no_grad()
def compute_token_heatmaps_for_atoms_on_sample(
    sae: Any,
    sample_or_batch: Any,
    atom_ids: Sequence[int],
    *,
    device: torch.device | str,
    grid_size: Tuple[int, int] | None = None,
    ignore_cls_token: bool = True,
    encoder: Any | None = None,
    features_list: Any | None = None,
) -> Dict[int, np.ndarray]:
    """Generate token-response heatmaps for multiple atoms on one sample (B=1).

    Args:
        sample_or_batch: Single sample (expected B=1), in feature or image mode.
        atom_ids: Atom IDs to compute.
        device, grid_size, ignore_cls_token, encoder, features_list: Recovery settings.

    Returns:
        Dict[int, np.ndarray]: {atom_id: (H,W) heatmap}
    """
    x, _ = build_tokens_from_batch(sample_or_batch, device=device, encoder=encoder, features_list=features_list)
    assert x.shape[0] == 1, "sample_or_batch は単一サンプル（B=1）を想定します。"
    z = sae.encode(x)  # (1,T,C)
    T = int(z.shape[1])
    atom_ids = [int(a) for a in atom_ids]

    if ignore_cls_token:
        z_tok = z[:, 1:, :]  # (1,T-1,C)
    else:
        z_tok = z  # (1,T,C)

    # Determine the heatmap shape.
    patch_T = int(z_tok.shape[1])
    if grid_size is None:
        g = int(np.sqrt(patch_T))
        assert g * g == patch_T, "grid_size を明示指定してください。"
        H = W = g
    else:
        H, W = grid_size
        assert H * W == patch_T, "grid_size とトークン数が一致しません。"

    # Slice vectors for the specified atoms and reshape them to 2D.
    out: Dict[int, np.ndarray] = {}
    for a in atom_ids:
        vec = z_tok[0, :, a].detach().to("cpu").numpy()
        out[a] = vec.reshape(H, W).astype(np.float32, copy=False)
    return out

@torch.no_grad()
def top_samples_for_atom_streaming(
    sae: Any,
    data: Iterable[Any],
    atom_ids: Sequence[int],
    *,
    device: torch.device | str,
    topn: int = 20,
    reduce: str = "max",
    encoder: Any | None = None,
    features_list: Any | None = None,
    max_batches: int | None = None,
) -> Dict[int, list[Dict[str, Any]]]:
    """Extract top samples (topn) for each target atom in streaming mode.
    This is constant-memory and single-pass. sample_id is assigned in stream order.
    """
    import heapq
    sae = sae.eval()
    device = torch.device(device)
    atom_ids = [int(a) for a in atom_ids]
    C = int(sae.hidden_dim)
    for a in atom_ids:
        assert 0 <= a < C
    # min-heap per atom_id
    heaps: Dict[int, list[tuple[float, int]]] = {a: [] for a in atom_ids}
    next_sid = 0

    for b_idx, batch in enumerate(data):
        if max_batches is not None and b_idx >= max_batches:
            break
        x, _ = build_tokens_from_batch(batch, device=device, encoder=encoder, features_list=features_list)
        z = sae.encode(x)  # (B,T,C)
        B, T, C_ = z.shape
        z_cpu = z.detach().to("cpu")
        for a in atom_ids:
            za = z_cpu[..., a]  # (B,T)
            if reduce == "max":
                scores = za.max(dim=1).values  # (B,)
            elif reduce == "sum":
                scores = za.sum(dim=1)  # (B,)
            elif reduce == "mean":
                scores = za.mean(dim=1)  # (B,)
            else:
                raise ValueError("reduce must be one of {'max','sum','mean'}")
            arr = scores.numpy().tolist()
            h = heaps[a]
            for i, sc in enumerate(arr):
                sid = next_sid + i
                item = (float(sc), int(sid))
                if len(h) < topn:
                    heapq.heappush(h, item)
                else:
                    if item[0] > h[0][0]:
                        heapq.heapreplace(h, item)
        next_sid += B

    # format output descending
    out: Dict[int, list[Dict[str, Any]]] = {}
    for a, h in heaps.items():
        h_sorted = sorted(h, key=lambda x: -x[0])
        out[a] = [{"sample_id": sid, "score": sc} for sc, sid in h_sorted]
    return out


@torch.no_grad()
def compute_token_heatmap_for_atom(
    sae: Any,
    sample_or_batch: Any,
    atom_id: int,
    *,
    device: torch.device | str,
    grid_size: Tuple[int, int] | None = None,
    ignore_cls_token: bool = True,
    encoder: Any | None = None,
    features_list: Any | None = None,
) -> np.ndarray:
    """Compute the token-response heatmap for a given atom from one sample or a B=1 batch."""
    x, _ = build_tokens_from_batch(sample_or_batch, device=device, encoder=encoder, features_list=features_list)
    assert x.shape[0] == 1, "sample_or_batch は単一サンプル（B=1）を想定します。"
    z = sae.encode(x)  # (1,T,C)
    T = int(z.shape[1])
    if ignore_cls_token:
        z_tok = z[:, 1:, atom_id]  # (1,T-1)
    else:
        z_tok = z[:, :, atom_id]   # (1,T)
    vec = z_tok.squeeze(0).detach().to("cpu").numpy()

    patch_T = vec.shape[0]
    if grid_size is None:
        g = int(np.sqrt(patch_T))
        assert g * g == patch_T, "grid_size を明示指定してください。"
        H = W = g
    else:
        H, W = grid_size
        assert H * W == patch_T, "grid_size とトークン数が一致しません。"
    heat = vec.copy()
    return heat.reshape(H, W).astype(np.float32, copy=False)

def top_samples_for_atom_from_index(
    index: Dict[int, Dict[str, np.ndarray]],
    atom_id: int,
    *,
    topn: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top samples with the most hits from the index."""
    ev = index.get(int(atom_id))
    if ev is None or ev["sample"].size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    uniq, cnt = np.unique(ev["sample"], return_counts=True)
    order = np.argsort(-cnt)[:topn]
    return uniq[order], cnt[order]


def heatmap_for_atom_on_sample_from_index(
    index: Dict[int, Dict[str, np.ndarray]],
    atom_id: int,
    sample_id: int,
    *,
    grid_size: Tuple[int, int] | None = None,
    ignore_cls_token: bool = True,
) -> np.ndarray | None:
    """Recover token responses for (atom_id, sample_id) from the index as a 2D heatmap."""
    ev = index.get(int(atom_id))
    if ev is None or ev["sample"].size == 0:
        return None
    m = (ev["sample"] == int(sample_id))
    if not np.any(m):
        return None
    tok = ev["token"][m].copy()
    val = ev["value"][m].astype(np.float32)

    if ignore_cls_token:
        mask = tok > 0
        tok, val = tok[mask], val[mask]
        tok = tok - 1

    Tm1 = int(tok.max()) + 1 if tok.size > 0 else 0
    if grid_size is None:
        g = int(np.sqrt(Tm1))
        assert g > 0 and g * g == Tm1, "grid_size を (H,W) で指定してください。"
        H = W = g
    else:
        H, W = grid_size
        assert H * W >= Tm1, "grid_size とトークン数が一致しません。"

    heat = np.zeros((H * W,), dtype=np.float32)
    if tok.size > 0:
        np.maximum.at(heat, tok, val)
    return heat.reshape(H, W)


def token_heatmap_for_atom_on_sample(
    rec_out: Dict[str, Any],
    atom_idx: int,
    sample_id: int,
    *,
    grid_size: Tuple[int, int] | None = None,
    ignore_cls_token: bool = True,
) -> np.ndarray:
    """Recover a 2D heatmap of atom responses for a sample from recording output."""
    T = rec_out["T"]
    assert T is not None, "Tが未確定です。"
    patch_T = T - 1 if ignore_cls_token else T

    if grid_size is None:
        g = int(np.sqrt(patch_T))
        assert g * g == patch_T, "grid_size を明示指定してください。"
        H = W = g
    else:
        H, W = grid_size
        assert H * W == patch_T, "grid_size とトークン数が一致しません。"

    ev = rec_out["events"]
    sid = ev["sample"]; tid = ev["token"]; aid = ev["atom"]; val = ev["value"]
    m = (sid == sample_id) & (aid == atom_idx)

    heat = np.zeros((patch_T,), dtype=np.float32)
    if m.sum() > 0:
        t_sel = tid[m]; v_sel = val[m]
        if ignore_cls_token:
            mask_patch = t_sel > 0
            t_sel = t_sel[mask_patch] - 1
            v_sel = v_sel[mask_patch]
        np.maximum.at(heat, t_sel, v_sel)

    return heat.reshape(H, W)

