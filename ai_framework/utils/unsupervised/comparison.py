"""Model comparison and dispatching for unsupervised algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ai_framework.models.unsupervised import Cluster

CLUSTERING_COMPARISON_ALGORITHMS = ("K-Means", "K-Medoids", "HAC")
ALGORITMOS_CLUSTERING_COMPARACION = CLUSTERING_COMPARISON_ALGORITHMS


def dispatch_algorithm(cluster: Cluster, algoritmo: str = "KMEANS", **kwargs) -> Any:
    """Dispatch execution of the requested algorithm on the clustering instance."""
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


def compare_clustering_algorithms(
    cluster: Cluster,
    n_clusters: int = 3,
    algoritmos: tuple[str, ...] | list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run and compare multiple clustering algorithms on the same dataset."""
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

# Backward compatibility aliases
ALGORITMOS_CLUSTERING_COMPARACION = CLUSTERING_COMPARISON_ALGORITHMS
despachar_algoritmo = dispatch_algorithm
comparar_algoritmos_clustering = compare_clustering_algorithms
