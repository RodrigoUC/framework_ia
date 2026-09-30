"""Modelos no supervisados: agrupamiento (incluye ACP, t-SNE y UMAP)."""

from .agrupamiento import Cluster, DependenciaOpcionalError
from .base import NoSupervisado

__all__ = [
    "NoSupervisado",
    "Cluster",
    "DependenciaOpcionalError",
]
