"""Unsupervised learning models: clustering and dimensionality reduction."""

from __future__ import annotations

from .clustering import Cluster, Clustering
from .unsupervised import DependenciaOpcionalError, NoSupervisado, OptionalDependencyError, Unsupervised

__all__ = [
    "Unsupervised",
    "Clustering",
    "OptionalDependencyError",
    "NoSupervisado",
    "Cluster",
    "DependenciaOpcionalError",
]
