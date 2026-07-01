"""Facade for SAE analysis utilities.

The thin :class:`SAEAnalysis` front-end delegates to helper modules for:

- dictionary visualization/embedding
- activation recording
- reconstruction/activation metrics
- inverted index utilities
- anomaly detection helpers

Existing code can continue to use :class:`SAEAnalysis` directly.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple
from pathlib import Path as _PathType

import numpy as np
import torch

from ..sae import SAE
from . import viz, metrics, indexing, anomaly
from .recording import RecordingSession, start_recording, record_batch, finalize_recording


class SAEAnalysis:
    """High-level facade over analysis helpers for a single SAE.

    This class exposes a stable API that groups multiple analysis workflows:

    - dictionary embedding and plotting
    - activation event recording
    - reconstruction and activation metrics
    - inverted-index based search over atoms/samples
    - anomaly scoring utilities
    - streaming overlay visualization

    Internally, each method delegates to a specialized helper module.
    """
    def __init__(self, sae: SAE, device: torch.device | str):
        self.sae = sae.to(device)
        self.device = torch.device(device)
        self._rec: RecordingSession | None = None

    @torch.no_grad()
    def _dictionary(self) -> np.ndarray:
        d = self.sae.decoder.dictionary.matrix  # [C, D]
        return d.detach().to("cpu").numpy()

    # --- visualization / embedding ---
    def embed_dictionary(
        self,
        method: Literal["pca", "tsne", "umap"] = "pca",
        metric: Literal["euclidean", "cosine"] = "euclidean",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs,
    ) -> np.ndarray:
        """Embed the dictionary matrix (C, D) into a low-dimensional space."""
        return viz.embed_dictionary(
            self.sae,
            method=method,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )

    def plot_dictionary(
        self,
        method: Literal["pca", "tsne", "umap"] = "pca",
        metric: Literal["euclidean", "cosine"] = "euclidean",
        labels: Optional[Sequence[int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (6, 6),
        point_size: float = 12.0,
        alpha: float = 0.8,
        **kwargs,
    ) -> Tuple[np.ndarray, Any]:
        """Plot a scatter plot of embedded dictionary atoms."""
        return viz.plot_dictionary(
            self.sae,
            method=method,
            metric=metric,
            labels=labels,
            title=title,
            save_path=save_path,
            show=show,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            **kwargs,
        )

    # --- recording session ---
    @torch.no_grad()
    def start_recording(
        self,
        *,
        threshold: float = 0.0,
        topk: int | None = 8,
        per_class: bool = False,
    ) -> None:
        """Start a recording session for sparse activation events.

        Args:
            threshold: Activation threshold used to mark an event (`z > threshold`).
            topk: Optional top-k event logging per token. Use `None` to disable.
            per_class: If True, keep class-wise online counters in addition to global stats.
        """
        C = self.sae.hidden_dim
        self._rec = start_recording(C=C, threshold=threshold, topk=topk, per_class=per_class)

    @torch.no_grad()
    def record_batch(
        self,
        batch: Any,
        *,
        encoder: Any | None = None,
        features_list: Any | None = None,
        class_extractor: Any | None = None,
    ) -> None:
        """Append one batch of events to the active recording session.

        The method supports both feature-mode and image-mode batches and delegates
        token extraction/encoding details to the recording backend.
        """
        assert self._rec is not None, "start_recording() must be called first."
        record_batch(
            self._rec,
            batch,
            sae=self.sae,
            device=self.device,
            encoder=encoder,
            features_list=features_list,
            class_extractor=class_extractor,
        )

    def finalize_recording(self) -> Dict[str, Any]:
        """Finalize the current recording session.

        Returns:
            Dictionary containing sparse event arrays and summary statistics.
        """
        assert self._rec is not None, "start_recording() must be called first."
        return finalize_recording(self._rec)

    # --- utilities on rec/index ---
    def token_heatmap_for_atom_on_sample(
        self,
        rec_out: Dict[str, Any],
        atom_idx: int,
        sample_id: int,
        *,
        grid_size: Tuple[int, int] | None = None,
        ignore_cls_token: bool = True,
    ) -> np.ndarray:
        """Reconstruct a 2D heatmap of atom activations for a given sample."""
        return indexing.token_heatmap_for_atom_on_sample(
            rec_out, atom_idx, sample_id, grid_size=grid_size, ignore_cls_token=ignore_cls_token
        )

    def top_samples_for_atom_from_index(
        self,
        index: Dict[int, Dict[str, np.ndarray]],
        atom_id: int,
        *,
        topn: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-N samples with strongest responses for a given atom from an index."""
        return indexing.top_samples_for_atom_from_index(index, atom_id, topn=topn)

    def heatmap_for_atom_on_sample_from_index(
        self,
        index: Dict[int, Dict[str, np.ndarray]],
        atom_id: int,
        sample_id: int,
        *,
        grid_size: Tuple[int, int] | None = None,
        ignore_cls_token: bool = True,
    ) -> np.ndarray | None:
        """Reconstruct a token-level heatmap from an inverted index entry."""
        return indexing.heatmap_for_atom_on_sample_from_index(
            index, atom_id, sample_id, grid_size=grid_size, ignore_cls_token=ignore_cls_token
        )

    @torch.no_grad()
    def show_sample_with_heatmap_from_index(
        self,
        sample_id: int,
        sample_meta: Dict[int, Dict[str, Any]],
        heatmap: np.ndarray,
        *,
        root: str | _PathType = "../../datasets/mvtec_anomaly_detection",
        alpha: float = 0.45,
        cmap: str = "jet",
        figsize: Tuple[int, int] = (6, 6),
        title: str | None = None,
    ):
        """Load an image and overlay a heatmap, returning a matplotlib Figure."""
        return viz.show_sample_with_heatmap_from_index(
            sample_id, sample_meta, heatmap, root=root, alpha=alpha, cmap=cmap, figsize=figsize, title=title
        )

    @torch.no_grad()
    def preview_top_samples_for_atom_from_index(
        self,
        index: Dict[int, Dict[str, np.ndarray]],
        sample_meta: Dict[int, Dict[str, Any]],
        atom_id: int,
        *,
        topn: int = 6,
        grid_size: Tuple[int, int] | None = None,
        ignore_cls_token: bool = True,
        alpha: float = 0.45,
        cmap: str = "jet",
        cols: int = 3,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """Quickly preview top-N images for an atom with overlays."""
        return viz.preview_top_samples_for_atom_from_index(
            index=index,
            sample_meta=sample_meta,
            atom_id=atom_id,
            topn=topn,
            grid_size=grid_size,
            ignore_cls_token=ignore_cls_token,
            alpha=alpha,
            cmap=cmap,
            cols=cols,
            figsize=figsize,
        )

    # --- metrics ---
    @torch.no_grad()
    def reconstruction_quality(
        self,
        data: Iterable[Any],
        *,
        encoder: Any | None = None,
        features_list: Any | None = None,
        max_batches: int | None = None,
        per_class: bool = False,
        class_extractor: Any | None = None,
        cls_only: bool = False,
        eps: float = 1e-8,
    ) -> Dict[str, Any]:
        """Aggregate reconstruction quality metrics in a streaming manner.

        Metrics include R^2, MSE, and mean cosine similarity.
        """
        return metrics.reconstruction_quality(
            self.sae,
            data,
            device=self.device,
            encoder=encoder,
            features_list=features_list,
            max_batches=max_batches,
            per_class=per_class,
            class_extractor=class_extractor,
            cls_only=cls_only,
            eps=eps,
        )

    @torch.no_grad()
    def activation_stats(
        self,
        data: Iterable[Any],
        *,
        encoder: Any | None = None,
        features_list: Any | None = None,
        threshold: float = 0.0,
        topk: int | None = None,
        max_batches: int | None = None,
        per_class: bool = False,
        class_extractor: Any | None = None,
        save_events: bool = False,
        save_meta: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate activation statistics for latent code `z`.

        Includes usage rate, active count, top-k rate (optional), and class-wise
        breakdowns when requested.
        """
        return metrics.activation_stats(
            self.sae,
            data,
            device=self.device,
            encoder=encoder,
            features_list=features_list,
            threshold=threshold,
            topk=topk,
            max_batches=max_batches,
            per_class=per_class,
            class_extractor=class_extractor,
            save_events=save_events,
            save_meta=save_meta,
        )

    # --- anomaly ---
    @torch.no_grad()
    def anomaly_with_atoms(
        self,
        data: Iterable[Any],
        anomaly_atoms: Sequence[int],
        *,
        encoder: Any | None = None,
        features_list: Any | None = None,
        agg: Literal["max", "lse", "mean"] = "max",
        temperature: float = 1.0,
        per_token_map: bool = False,
        grid_size: Tuple[int, int] | None = None,
        max_batches: int | None = None,
    ) -> Dict[str, Any]:
        """Compute similarity-based anomaly scores using selected atom IDs."""
        return anomaly.anomaly_with_atoms(
            self.sae,
            data,
            anomaly_atoms,
            device=self.device,
            encoder=encoder,
            features_list=features_list,
            agg=agg,
            temperature=temperature,
            per_token_map=per_token_map,
            grid_size=grid_size,
            max_batches=max_batches,
        )

    def top_images_for_atom(
        self,
        rec_out: Dict[str, Any],
        atom_idx: int,
        *,
        topn: int = 20,
        reduce: str = "max",
    ) -> list[Dict[str, Any]]:
        """Return top-N images with strongest responses for one atom."""
        return anomaly.top_images_for_atom(rec_out, atom_idx, topn=topn, reduce=reduce)

    # --- inverted index ---
    @torch.no_grad()
    def build_atom_inverted_index(
        self,
        data: Iterable[Any],
        atom_ids: Sequence[int],
        *,
        threshold: float = 0.0,
        encoder: Any | None = None,
        features_list: Any | None = None,
        exclude_cls_token: bool = True,
        max_batches: int | None = None,
    ) -> tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, Any]]]:
        """Build an inverted index of sparse events for the selected atoms."""
        return indexing.build_atom_inverted_index(
            self.sae,
            data,
            atom_ids,
            device=self.device,
            threshold=threshold,
            encoder=encoder,
            features_list=features_list,
            exclude_cls_token=exclude_cls_token,
            max_batches=max_batches,
        )

    # --- streaming ranking / on-demand heatmap ---
    @torch.no_grad()
    def top_samples_for_atom_streaming(
        self,
        data: Iterable[Any],
        atom_ids: Sequence[int],
        *,
        topn: int = 20,
        reduce: Literal["max", "sum", "mean"] = "max",
        encoder: Any | None = None,
        features_list: Any | None = None,
        max_batches: int | None = None,
    ) -> Dict[int, list[Dict[str, Any]]]:
        """Extract top samples per atom from a streaming pass over data."""
        return indexing.top_samples_for_atom_streaming(
            self.sae,
            data,
            atom_ids,
            device=self.device,
            topn=topn,
            reduce=reduce,
            encoder=encoder,
            features_list=features_list,
            max_batches=max_batches,
        )

    @torch.no_grad()
    def compute_token_heatmap_for_atom(
        self,
        sample_or_batch: Any,
        atom_id: int,
        *,
        grid_size: Tuple[int, int] | None = None,
        ignore_cls_token: bool = True,
        encoder: Any | None = None,
        features_list: Any | None = None,
    ) -> np.ndarray:
        """Compute a token-level activation heatmap for one sample and atom."""
        return indexing.compute_token_heatmap_for_atom(
            self.sae,
            sample_or_batch,
            atom_id,
            device=self.device,
            grid_size=grid_size,
            ignore_cls_token=ignore_cls_token,
            encoder=encoder,
            features_list=features_list,
        )


    @torch.no_grad()
    def show_top_overlays_for_atom_streaming(
        self,
        data: Iterable[Any],
        atom_id: int,
        *,
        topn: int = 12,
        cols: int = 4,
        grid_size: Tuple[int, int] | None = None,
        ignore_cls_token: bool = True,
        alpha: float = 0.45,
        cmap: str = "jet",
        figsize: Tuple[int, int] = (12, 9),
        encoder: Any | None = None,
        features_list: Any | None = None,
        root: str | _PathType = "../../datasets/mvtec_anomaly_detection",
    ) -> None:
        """Render tiled image/heatmap overlays for top-N samples of one atom.

        The method performs a streaming ranking pass, then re-reads only selected
        samples to generate overlays.
        """
        from math import ceil
        from pathlib import Path as _Path
        from PIL import Image
        from .deps import require_matplotlib_pyplot

        plt = require_matplotlib_pyplot()

        # 1) Extract top-ranked samples in a streaming pass.
        tops_dict = indexing.top_samples_for_atom_streaming(
            self.sae,
            data,
            [atom_id],
            device=self.device,
            topn=topn,
            reduce="max",
            encoder=encoder,
            features_list=features_list,
            max_batches=None,
        )
        tops = tops_dict.get(int(atom_id), [])
        if len(tops) == 0:
            # No positive responses found for this atom.
            return

        # 2) Collect target sample IDs while preserving rank order.
        wanted = [int(x.get("sample_id", -1)) for x in tops]
        wanted_set = set(wanted)

        # 3) Re-scan data and build overlay materials for the selected sample IDs.
        overlays: list[tuple[Any, np.ndarray, str]] = []  # (PIL.Image, heatmap, title)
        sid = 0
        from pathlib import Path as _Path
        from PIL import Image

        for batch in data:
            # Infer batch size from known batch formats.
            if hasattr(batch, "feats") and hasattr(batch.feats, "shape"):
                B = int(batch.feats.shape[0])
            elif hasattr(batch, "mid_feats") and isinstance(batch.mid_feats, list):
                B = int(batch.mid_feats[-1].shape[0])
            elif isinstance(batch, list):
                B = len(batch)
            else:
                # Skip unsupported batch structures.
                continue

            for i in range(B):
                cur_sid = sid + i
                if cur_sid not in wanted_set:
                    continue

                # Resolve image path from the batch item.
                img_rel: str | None = None
                if hasattr(batch, "img_path"):
                    if isinstance(batch.img_path, list):
                        img_rel = str(batch.img_path[i])
                    else:
                        img_rel = str(batch.img_path)
                elif isinstance(batch, list) and len(batch) > i and hasattr(batch[i], "img_path"):
                    img_rel = str(batch[i].img_path)

                if img_rel is None:
                    # Overlay cannot be created without an image path.
                    continue

                # Build a lightweight record object (feature mode first, then image mode).
                class _Rec:
                    pass
                rec = _Rec()
                rec.img_path = img_rel

                if hasattr(batch, "feats") and isinstance(getattr(batch, "mid_feats", None), list):
                    # Attach per-sample feats and last-layer mid-features.
                    rec.feats = batch.feats[i]
                    last_mid = batch.mid_feats[-1][i]  # (N+1, D)
                    rec.mid_feats = [last_mid]
                    # optional class name
                    if hasattr(batch, "cls_name"):
                        rec.cls_name = batch.cls_name[i] if isinstance(batch.cls_name, list) else batch.cls_name
                    # Load PIL image for visualization.
                    img = Image.open(_Path(root) / img_rel).convert("RGB")
                    rec.img = img  # image used for visualization only
                    heat = indexing.compute_token_heatmap_for_atom(
                        self.sae,
                        rec,
                        atom_id,
                        device=self.device,
                        grid_size=grid_size,
                        ignore_cls_token=ignore_cls_token,
                        encoder=None,  # not needed in feature mode
                        features_list=None,
                    )
                else:
                    # Fallback to image-mode heatmap computation.
                    img = Image.open(_Path(root) / img_rel).convert("RGB")
                    rec.img = img
                    heat = indexing.compute_token_heatmap_for_atom(
                        self.sae,
                        rec,
                        atom_id,
                        device=self.device,
                        grid_size=grid_size,
                        ignore_cls_token=ignore_cls_token,
                        encoder=encoder,
                        features_list=features_list,
                    )

                sc = next((t.get("score") for t in tops if int(t.get("sample_id", -1)) == cur_sid), None)
                title = f"atom={atom_id} sid={cur_sid}" + (f" s={sc:.3f}" if isinstance(sc, (int, float)) else "")
                overlays.append((img, heat, title, sc, cur_sid))

                if len(overlays) >= len(wanted):
                    break
            sid += B
            if len(overlays) >= len(wanted):
                break

        # 4) Sort by score and render tiled overlays.
        if len(overlays) == 0:
            return
        overlays.sort(key=lambda t: (t[3] if isinstance(t[3], (int, float)) else float("-inf")), reverse=True)
        n = min(len(overlays), topn)
        rows = ceil(n / max(1, cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [[axes]]  # type: ignore
        elif rows == 1:
            axes = [axes]  # type: ignore
        elif cols == 1:
            axes = [[ax] for ax in axes]  # type: ignore
        flat_axes = [ax for row in axes for ax in row]  # type: ignore

        for i, ax in enumerate(flat_axes):
            if i >= n:
                ax.axis("off")
                continue
            img, heat, title, _, _ = overlays[i]
            ax.imshow(img)
            ax.imshow(
                heat,
                cmap=cmap,
                alpha=alpha,
                interpolation="nearest",
                extent=(0, img.width, img.height, 0),
            )
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


    @torch.no_grad()
    def show_top_overlays_for_atoms_streaming(
        self,
        data: Iterable[Any],
        atom_ids: Sequence[int],
        *,
        topn: int = 12,
        per_class_topk: int | None = None,
        cols_per_atom: int = 4,
        layout: Literal["per_atom", "mega"] = "per_atom",
        atoms_per_row: int = 4,
        figsize: Tuple[int, int] = (12, 9),
        alpha: float = 0.45,
        cmap: str = "jet",
        grid_size: Tuple[int, int] | None = None,
        ignore_cls_token: bool = True,
        encoder: Any | None = None,
        features_list: Any | None = None,
        root: str | _PathType = "../../datasets/mvtec_anomaly_detection",
        save_dir: str | _PathType | None = None,
        show: bool = True,
        chunk_size: int | None = None,
        dpi: int = 120,
        save_individual_tiles: bool = False,
        save_classwise_tiles: bool = False,
    ) -> Dict[str, Any]:
        """Generate overlays for multiple atoms and return summary statistics.

        This method supports both per-atom and mega layout modes, optional
        per-class top-k preservation, and optional saving of individual/classwise tiles.
        """
        from collections import defaultdict
        from math import ceil
        from pathlib import Path as _Path
        from PIL import Image
        from .deps import require_matplotlib_pyplot

        plt = require_matplotlib_pyplot()

        atom_ids = [int(a) for a in atom_ids]
        if chunk_size is None or chunk_size <= 0:
            chunks = [atom_ids]
        else:
            chunks = [atom_ids[i:i+chunk_size] for i in range(0, len(atom_ids), chunk_size)]

        # Extract top-N samples for all atoms in one streaming pass.
        # If per_class_topk is enabled, also keep class-wise top-K heaps.
        tops_all: Dict[int, list[Dict[str, Any]]] = {}
        tops_by_class: Dict[int, Dict[str, list[Dict[str, Any]]]] = {}
        if per_class_topk:
            import heapq
            from .tokens import build_tokens_from_batch

            def _to_str(v: Any) -> Optional[str]:
                if v is None:
                    return None
                if hasattr(v, "item"):
                    try:
                        return str(v.item())
                    except Exception:
                        return None
                return str(v)

            def _extract_cls_names(batch: Any, B: int) -> list[Optional[str]]:
                if hasattr(batch, "cls_name"):
                    c = batch.cls_name
                    if isinstance(c, list):
                        if len(c) == B:
                            return [_to_str(x) for x in c]
                    if hasattr(c, "shape") and len(getattr(c, "shape", [])) > 0:
                        try:
                            if int(c.shape[0]) == B:
                                return [_to_str(c[i]) for i in range(B)]
                        except Exception:
                            pass
                    return [_to_str(c)] * B
                return [None] * B

            sae = self.sae.eval()
            device = torch.device(self.device)
            atom_ids = [int(a) for a in atom_ids]
            heaps_global: Dict[int, list[tuple[float, int]]] = {a: [] for a in atom_ids}
            heaps_by_class: Dict[int, Dict[str, list[tuple[float, int]]]] = {a: {} for a in atom_ids}
            next_sid = 0

            for batch in data:
                x, _ = build_tokens_from_batch(batch, device=device, encoder=encoder, features_list=features_list)
                z = sae.encode(x)  # (B,T,C)
                B = int(z.shape[0])
                z_cpu = z.detach().to("cpu")
                cls_names = _extract_cls_names(batch, B)

                for a in atom_ids:
                    za = z_cpu[..., a]  # (B,T)
                    scores = za.max(dim=1).values
                    arr = scores.numpy().tolist()
                    h_g = heaps_global[a]
                    h_by_cls = heaps_by_class[a]
                    for i, sc in enumerate(arr):
                        sid = next_sid + i
                        item = (float(sc), int(sid))
                        if len(h_g) < topn:
                            heapq.heappush(h_g, item)
                        else:
                            if item[0] > h_g[0][0]:
                                heapq.heapreplace(h_g, item)

                        cls = cls_names[i]
                        if cls is not None:
                            h_c = h_by_cls.setdefault(cls, [])
                            if len(h_c) < int(per_class_topk):
                                heapq.heappush(h_c, item)
                            else:
                                if item[0] > h_c[0][0]:
                                    heapq.heapreplace(h_c, item)

                next_sid += B

            for a, h in heaps_global.items():
                h_sorted = sorted(h, key=lambda x: -x[0])
                tops_all[a] = [{"sample_id": sid, "score": sc} for sc, sid in h_sorted]
            for a, by_cls in heaps_by_class.items():
                tops_by_class[a] = {}
                for cls, h in by_cls.items():
                    h_sorted = sorted(h, key=lambda x: -x[0])
                    tops_by_class[a][cls] = [{"sample_id": sid, "score": sc, "cls": cls} for sc, sid in h_sorted]
        else:
            tops_all = indexing.top_samples_for_atom_streaming(
                self.sae,
                data,
                atom_ids,
                device=self.device,
                topn=topn,
                reduce="max",
                encoder=encoder,
                features_list=features_list,
                max_batches=None,
            )

        # Keep only positive-score samples and cap at topn / per_class_topk.
        for a in atom_ids:
            tops_all[a] = [x for x in tops_all.get(a, []) if (x.get("score") or 0) > 0][:topn]
            if per_class_topk:
                by_cls = tops_by_class.get(a, {})
                if by_cls:
                    for cls in list(by_cls.keys()):
                        kept = [x for x in by_cls[cls] if (x.get("score") or 0) > 0][: int(per_class_topk)]
                        if kept:
                            by_cls[cls] = kept
                        else:
                            del by_cls[cls]

        # Build the set of required sample IDs for the second pass.
        classwise_orders: Dict[int, Dict[str, list[int]]] = {}
        wanted_per_atom: Dict[int, list[int]] = {}
        for a in atom_ids:
            sids = [int(x.get("sample_id", -1)) for x in tops_all.get(a, [])]
            if per_class_topk:
                by_cls = tops_by_class.get(a, {})
                if by_cls:
                    classwise_orders[a] = {}
                    for cls, lst in by_cls.items():
                        cls_sids = [int(x.get("sample_id", -1)) for x in lst]
                        classwise_orders[a][cls] = cls_sids
                        sids.extend(cls_sids)
            # preserve order while removing dups
            seen = set()
            dedup = []
            for sid in sids:
                if sid in seen:
                    continue
                seen.add(sid)
                dedup.append(sid)
            wanted_per_atom[a] = dedup
        wanted_set = set([sid for sids in wanted_per_atom.values() for sid in sids])

        # Build reusable overlay materials per sample ID.
        # Keep atom-wise containers so rank/order can be restored later.
        overlays_per_atom: Dict[int, list[tuple[Any, np.ndarray, str, Any, int, Optional[str]]]] = defaultdict(list)
        score_lookup: Dict[int, Dict[int, float]] = {a: {} for a in atom_ids}
        for a in atom_ids:
            for t in tops_all.get(a, []):
                try:
                    score_lookup[a][int(t.get("sample_id", -1))] = float(t.get("score"))
                except Exception:
                    pass
            if per_class_topk:
                for lst in tops_by_class.get(a, {}).values():
                    for t in lst:
                        sid = int(t.get("sample_id", -1))
                        if sid not in score_lookup[a]:
                            try:
                                score_lookup[a][sid] = float(t.get("score"))
                            except Exception:
                                pass

        sid = 0
        for batch in data:
            # Infer batch size.
            if hasattr(batch, "feats") and hasattr(batch.feats, "shape"):
                B = int(batch.feats.shape[0])
            elif hasattr(batch, "mid_feats") and isinstance(batch.mid_feats, list):
                B = int(batch.mid_feats[-1].shape[0])
            elif isinstance(batch, list):
                B = len(batch)
            else:
                continue

            # Collect required samples that appear in this batch.
            local_map: Dict[int, Dict[str, Any]] = {}
            for i in range(B):
                cur_sid = sid + i
                if cur_sid in wanted_set:
                    # Resolve image path.
                    img_rel: str | None = None
                    if hasattr(batch, "img_path"):
                        img_rel = str(batch.img_path[i]) if isinstance(batch.img_path, list) else str(batch.img_path)
                    elif isinstance(batch, list) and len(batch) > i and hasattr(batch[i], "img_path"):
                        img_rel = str(batch[i].img_path)
                    if img_rel is None:
                        continue

                    # Create a lightweight record for feature-mode or image-mode paths.
                    class _Rec: pass
                    rec = _Rec()
                    rec.img_path = img_rel
                    if hasattr(batch, "feats") and isinstance(getattr(batch, "mid_feats", None), list):
                        rec.feats = batch.feats[i]
                        last_mid = batch.mid_feats[-1][i]
                        rec.mid_feats = [last_mid]
                        if hasattr(batch, "cls_name"):
                            rec.cls_name = batch.cls_name[i] if isinstance(batch.cls_name, list) else batch.cls_name
                    else:
                        # Image-mode fallback: open PIL image.
                        rec.img = Image.open(_Path(root) / img_rel).convert("RGB")
                    # Resolve class name if available.
                    cls_val = None
                    if hasattr(batch, "cls_name"):
                        cls_val = batch.cls_name[i] if isinstance(batch.cls_name, list) else batch.cls_name
                        if hasattr(cls_val, "item"):
                            try:
                                cls_val = str(cls_val.item())
                            except Exception:
                                cls_val = None
                        else:
                            cls_val = str(cls_val)
                    local_map[cur_sid] = {"rec": rec, "img_rel": img_rel, "cls": cls_val}

            # Compute heatmaps for required sample IDs in this batch.
            for cur_sid, entry in local_map.items():
                rec = entry["rec"]
                # Determine which atoms need this sample ID.
                atoms_for_sid = [a for a, sids in wanted_per_atom.items() if cur_sid in sids]
                # Compute multiple atom heatmaps in one call.
                heatmaps = indexing.compute_token_heatmaps_for_atoms_on_sample(
                    self.sae,
                    rec,
                    atoms_for_sid,
                    device=self.device,
                    grid_size=grid_size,
                    ignore_cls_token=ignore_cls_token,
                    encoder=None if hasattr(rec, "feats") else encoder,
                    features_list=None if hasattr(rec, "feats") else features_list,
                )
                # Ensure an image object is available for overlay rendering.
                if hasattr(rec, "img"):
                    img = rec.img
                else:
                    img = Image.open(_Path(root) / entry["img_rel"]).convert("RGB")

                for a in atoms_for_sid:
                    sc = score_lookup.get(a, {}).get(cur_sid)
                    title = f"atom={a} sid={cur_sid}" + (f" s={sc:.3f}" if isinstance(sc, (int, float)) else "")
                    hm = heatmaps[a]
                    overlays_per_atom[a].append((img, hm, title, sc, cur_sid, entry.get("cls")))

            sid += B

        # Render and/or save overlays.
        from .deps import require_matplotlib_pyplot
        plt = require_matplotlib_pyplot()

        def draw_tile(fig, axes, overlays, cols):
            flat_axes = [ax for row in axes for ax in row]
            for i, ax in enumerate(flat_axes):
                if i >= len(overlays):
                    ax.axis("off")
                    continue
                img, heat, title, _, _, _ = overlays[i]
                ax.imshow(img)
                ax.imshow(heat, cmap=cmap, alpha=alpha, interpolation="nearest", extent=(0, img.width, img.height, 0))
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig.tight_layout()

        def _save_single_tile(img, heat, out_path):
            fig_tile, ax_tile = plt.subplots(1, 1, figsize=figsize)
            ax_tile.imshow(img)
            ax_tile.imshow(
                heat,
                cmap=cmap,
                alpha=alpha,
                interpolation="nearest",
                extent=(0, img.width, img.height, 0),
            )
            ax_tile.axis("off")
            fig_tile.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig_tile.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig_tile)

        # Sort overlays by score for statistics and rendering.
        per_atom_sorted: Dict[int, list[tuple[Any, np.ndarray, str, Any, int, Optional[str]]]] = {
            a: sorted(v, key=lambda t: (t[3] if isinstance(t[3], (int, float)) else float("-inf")), reverse=True)
            for a, v in overlays_per_atom.items()
        }
        # Build output statistics.
        stats_out: Dict[str, Any] = {"per_atom": {}, "global": {}}
        total_drawn = 0
        total_positive = 0
        global_classes_pos: set[str] = set()

        atoms_list = [a for a in atom_ids if len(overlays_per_atom.get(a, [])) > 0]
        if layout == "per_atom":
            for a in atoms_list:
                overlays = per_atom_sorted[a][:topn]
                n = len(overlays)
                if n == 0:
                    continue
                # per-atom stats
                pos_classes: list[str] = []
                num_pos = 0
                top_samples_meta = []
                for (img, heat, title, sc, sid_i, cls_name) in overlays:
                    top_samples_meta.append({"sample_id": int(sid_i), "score": float(sc) if sc is not None else float("nan"), "cls": cls_name})
                    if isinstance(sc, (int, float)) and sc > 0:
                        num_pos += 1
                        if isinstance(cls_name, str):
                            pos_classes.append(cls_name)
                            global_classes_pos.add(cls_name)
                class_counts = {}
                for c in pos_classes:
                    class_counts[c] = class_counts.get(c, 0) + 1
                stats_out["per_atom"][a] = {
                    "num_candidates": len(per_atom_sorted[a]),
                    "num_drawn": n,
                    "num_positive": num_pos,
                    "distinct_classes_positive": len(set(pos_classes)),
                    "class_counts_positive": class_counts,
                    "top_samples": top_samples_meta,
                }
                total_drawn += n
                total_positive += num_pos
                rows = ceil(n / max(1, cols_per_atom))
                fig, axes = plt.subplots(rows, cols_per_atom, figsize=figsize)
                if rows == 1 and cols_per_atom == 1:
                    axes = [[axes]]
                elif rows == 1:
                    axes = [axes]
                elif cols_per_atom == 1:
                    axes = [[ax] for ax in axes]
                draw_tile(fig, axes, overlays, cols_per_atom)
                if save_dir is not None:
                    _Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(_Path(save_dir) / f"atom_{a}.png", dpi=dpi)
                if save_individual_tiles and save_dir is not None:
                    atom_dir = _Path(save_dir) / f"atom_{a}"
                    atom_dir.mkdir(parents=True, exist_ok=True)
                    for r, (img, heat, _title, _sc, _sid_i, _cls_name) in enumerate(overlays):
                        _save_single_tile(img, heat, atom_dir / f"rank_{r}.png")
                if save_classwise_tiles and save_dir is not None and per_class_topk:
                    import re as _re

                    def _safe_cls_name(s: str) -> str:
                        s = s.strip()
                        s = _re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
                        return s if s else "unknown"

                    cls_order = classwise_orders.get(a, {})
                    if cls_order:
                        base_dir = _Path(save_dir) / f"atom_{a}" / "by_class"
                        base_dir.mkdir(parents=True, exist_ok=True)
                        overlays_by_sid = {int(sid_i): (img, heat) for (img, heat, _title, _sc, sid_i, _cls_name) in overlays_per_atom[a]}
                        for cls_name, sid_list in cls_order.items():
                            if not sid_list:
                                continue
                            cls_dir = base_dir / _safe_cls_name(cls_name)
                            cls_dir.mkdir(parents=True, exist_ok=True)
                            for r, sid in enumerate(sid_list):
                                item = overlays_by_sid.get(int(sid))
                                if item is None:
                                    continue
                                img, heat = item
                                _save_single_tile(img, heat, cls_dir / f"rank_{r}.png")
                if show:
                    plt.show()
                else:
                    plt.close(fig)
        else:  # mega
            # Split atoms across figures when using mega layout.
            if atoms_per_row <= 0:
                atoms_per_row = 4
            # Partition atoms by chunk.
            atoms_chunks = [atoms_list[i:i+ (chunk_size or len(atoms_list))] for i in range(0, len(atoms_list), (chunk_size or len(atoms_list)))]
            for idx_chunk, atoms_chunk in enumerate(atoms_chunks):
                if len(atoms_chunk) == 0:
                    continue
                rows = len(atoms_chunk)
                cols = topn
                fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], max(figsize[1], rows * figsize[1] / max(1, atoms_per_row))))
                if rows == 1 and cols == 1:
                    axes = [[axes]]
                elif rows == 1:
                    axes = [axes]
                elif cols == 1:
                    axes = [[ax] for ax in axes]
                # Draw each chunk.
                for r, a in enumerate(atoms_chunk):
                    overlays = per_atom_sorted[a][:topn]
                    # Build per-atom stats if not already created.
                    if a not in stats_out["per_atom"]:
                        pos_classes: list[str] = []
                        num_pos = 0
                        top_samples_meta = []
                        for (img, heat, title, sc, sid_i, cls_name) in overlays:
                            top_samples_meta.append({"sample_id": int(sid_i), "score": float(sc) if sc is not None else float("nan"), "cls": cls_name})
                            if isinstance(sc, (int, float)) and sc > 0:
                                num_pos += 1
                                if isinstance(cls_name, str):
                                    pos_classes.append(cls_name)
                                    global_classes_pos.add(cls_name)
                        class_counts = {}
                        for c in pos_classes:
                            class_counts[c] = class_counts.get(c, 0) + 1
                        stats_out["per_atom"][a] = {
                            "num_candidates": len(per_atom_sorted[a]),
                            "num_drawn": len(overlays),
                            "num_positive": num_pos,
                            "distinct_classes_positive": len(set(pos_classes)),
                            "class_counts_positive": class_counts,
                            "top_samples": top_samples_meta,
                        }
                        total_drawn += len(overlays)
                        total_positive += num_pos
                    for c in range(cols):
                        ax = axes[r][c]
                        if c < len(overlays):
                            img, heat, title, _, _, _ = overlays[c]
                            ax.imshow(img)
                            ax.imshow(heat, cmap=cmap, alpha=alpha, interpolation="nearest", extent=(0, img.width, img.height, 0))
                            ax.set_title(title, fontsize=8)
                        ax.axis("off")
                fig.tight_layout()
                if save_dir is not None:
                    _Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(_Path(save_dir) / f"atoms_mega_{idx_chunk:03d}.png", dpi=dpi)
                if show:
                    plt.show()
                else:
                    plt.close(fig)

        # Final global statistics.
        stats_out["global"] = {
            "atoms_processed": len(atoms_list),
            "total_drawn": int(total_drawn),
            "total_positive": int(total_positive),
            "distinct_classes_positive": len(global_classes_pos),
        }
        return stats_out
