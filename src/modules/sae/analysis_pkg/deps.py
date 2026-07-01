"""Lazy import helpers for external dependencies with install guidance when missing."""
from __future__ import annotations

class NotInstalledError(ImportError):
    """Raised when an optional dependency is not installed."""
    pass


def require_matplotlib_pyplot():
    """Lazy-import and return matplotlib.pyplot, or raise with install guidance."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise NotInstalledError(
            "matplotlib が必要です。`pip install matplotlib` を実行してください。"
        ) from e
    return plt


def require_sklearn_PCA():
    """Lazy-import and return sklearn.decomposition.PCA."""
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as e:
        raise NotInstalledError(
            "scikit-learn が必要です。`pip install scikit-learn` を実行してください。"
        ) from e
    return PCA


def require_sklearn_TSNE():
    """Lazy-import and return sklearn.manifold.TSNE."""
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as e:
        raise NotInstalledError(
            "scikit-learn が必要です。`pip install scikit-learn` を実行してください。"
        ) from e
    return TSNE


def require_umap_UMAP():
    """Lazy-import and return umap.UMAP."""
    try:
        from umap import UMAP  # type: ignore
    except Exception as e:
        raise NotInstalledError(
            "umap-learn が必要です。`pip install umap-learn` を実行してください。"
        ) from e
    return UMAP

