"""Dictionary embedding/visualization and image-plus-heatmap display helpers."""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import torch

from .deps import (
    require_matplotlib_pyplot,
    require_sklearn_PCA,
    require_sklearn_TSNE,
    require_umap_UMAP,
)
from .indexing import top_samples_for_atom_from_index, heatmap_for_atom_on_sample_from_index


def embed_dictionary(
    sae: Any,
    method: str = "pca",
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Embed the dictionary matrix (C, D) with PCA/TSNE/UMAP."""
    X = sae.decoder.dictionary.matrix.detach().to("cpu").numpy()
    m = method.lower()
    if m == "pca":
        PCA = require_sklearn_PCA()
        emb = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    elif m == "tsne":
        TSNE = require_sklearn_TSNE()
        emb = TSNE(
            n_components=n_components,
            metric=metric,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            **kwargs,
        ).fit_transform(X)
    elif m == "umap":
        UMAP = require_umap_UMAP()
        emb = UMAP(n_components=n_components, metric=metric, random_state=random_state, **kwargs).fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return emb


def plot_dictionary(
    sae: Any,
    method: str = "pca",
    metric: str = "euclidean",
    labels: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (6, 6),
    point_size: float = 12.0,
    alpha: float = 0.8,
    **kwargs,
) -> tuple[np.ndarray, Any]:
    """Plot a 2D scatter plot of the dictionary embedding."""
    plt = require_matplotlib_pyplot()
    emb = embed_dictionary(sae, method=method, metric=metric, n_components=2, **kwargs)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=point_size, alpha=alpha)
    else:
        labels = np.asarray(labels)
        assert len(labels) == emb.shape[0], "labels length must equal #atoms"
        for lab in np.unique(labels):
            idx = labels == lab
            ax.scatter(emb[idx, 0], emb[idx, 1], s=point_size, alpha=alpha, label=str(lab))
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_title(title or f"SAE Dictionary ({method.upper()})")
    ax.grid(True, linestyle="--", alpha=0.2)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return emb, fig


def show_sample_with_heatmap_from_index(
    sample_id: int,
    sample_meta: Dict[int, Dict[str, Any]],
    heatmap: np.ndarray,
    *,
    root: str | "Path" = "../../datasets/mvtec_anomaly_detection",
    alpha: float = 0.45,
    cmap: str = "jet",
    figsize: Tuple[int, int] = (6, 6),
    title: str | None = None,
) -> Any:
    """Return a figure with a heatmap overlaid on the sample image."""
    from pathlib import Path
    from PIL import Image
    plt = require_matplotlib_pyplot()

    img_path = sample_meta.get(int(sample_id), {}).get("img_path", None)
    if img_path is None:
        raise ValueError("img_path が sample_meta にありません。record時に保存してください。")
    img = Image.open(Path(root) / img_path).convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig


def preview_top_samples_for_atom_from_index(
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
) -> list[Any]:
    """Preview top samples for the specified atom as image-plus-heatmap figures."""
    sids, _ = top_samples_for_atom_from_index(index, atom_id, topn=topn)
    figs: list[Any] = []
    for sid in sids:
        hm = heatmap_for_atom_on_sample_from_index(
            index, atom_id, int(sid),
            grid_size=grid_size,
            ignore_cls_token=ignore_cls_token,
        )
        if hm is None:
            continue
        title = f"atom={atom_id}, sample_id={int(sid)}"
        fig = show_sample_with_heatmap_from_index(
            int(sid), sample_meta, hm, alpha=alpha, cmap=cmap, figsize=figsize, title=title
        )
        figs.append(fig)
    return figs

