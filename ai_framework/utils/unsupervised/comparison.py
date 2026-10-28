"""Comparación de modelos y despacho de algoritmos no supervisados."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ai_framework.models.unsupervised import Cluster

ALGORITMOS_CLUSTERING_COMPARACION = ("K-Means", "K-Medoids", "HAC")


def despachar_algoritmo(cluster: Cluster, algoritmo: str = "KMEANS", **kwargs) -> Any:
    """Despacha la ejecución del algoritmo solicitado en la instancia de clustering."""
    metodos = {
        "KMEANS": cluster.K_means,
        "KMEDOIDS": cluster.K_medoids,
        "HAC": cluster.HAC,
        "ACP": cluster.ACP,
        "TSNE": cluster.TSNE,
        "UMAP": cluster.UMAP,
    }
    clave = algoritmo.upper().replace("-", "").replace("_", "")
    if clave not in metodos:
        raise ValueError(f"Algoritmo no soportado para despacho: {algoritmo}")
    return metodos[clave](**kwargs)


def comparar_algoritmos_clustering(
    cluster: Cluster,
    n_clusters: int = 3,
    algoritmos: tuple[str, ...] | list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Ejecuta y compara múltiples algoritmos de agrupamiento sobre el mismo conjunto de datos."""
    candidatos = algoritmos if algoritmos is not None else ALGORITMOS_CLUSTERING_COMPARACION
    filas = []

    for algo in candidatos:
        resultado = despachar_algoritmo(cluster, algo, n_clusters=n_clusters, **kwargs)
        filas.append(
            {
                "algoritmo": resultado.algoritmo,
                "numero_clusters": resultado.etiquetas.nunique(),
                "silhouette": resultado.silhouette,
                "inercia": resultado.inercia,
            }
        )

    return pd.DataFrame(filas)
