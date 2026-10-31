"""Unsupervised learning models: clustering and dimensionality reduction."""

from __future__ import annotations

from .clustering import Clustering
from .unsupervised import OptionalDependencyError, Unsupervised


def __getattr__(name: str):
    if name in ("NoSupervisado", "Cluster", "DependenciaOpcionalError"):
        import ai_framework.utils.aliases as aliases
        return getattr(aliases, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Unsupervised",
    "Clustering",
    "OptionalDependencyError",
    "NoSupervisado",
    "Cluster",
    "DependenciaOpcionalError",
]
