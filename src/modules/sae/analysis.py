"""Compatibility shim that re-exports :class:`SAEAnalysis`.

The implementation now lives under :mod:`modules.sae.analysis_pkg`, but
existing imports from :mod:`modules.sae.analysis` continue to work.
"""
from .analysis_pkg.facade import SAEAnalysis

__all__ = ["SAEAnalysis"]


