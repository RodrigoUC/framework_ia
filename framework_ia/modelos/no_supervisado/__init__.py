"""Modelos no supervisados: agrupamiento (K-Means, K-Medoids, HAC) y proyecciones."""

from .agrupamiento import Cluster
from .no_supervisados import DependenciaOpcionalError, NoSupervisado

__all__ = [
    "NoSupervisado",
    "Cluster",
    "DependenciaOpcionalError",
]
