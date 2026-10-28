"""Utilidades especializadas para análisis no supervisado y clustering."""

from .comparacion import (
    ALGORITMOS_CLUSTERING_COMPARACION,
    comparar_algoritmos_clustering,
    despachar_algoritmo,
)

__all__ = [
    "ALGORITMOS_CLUSTERING_COMPARACION",
    "despachar_algoritmo",
    "comparar_algoritmos_clustering",
]
