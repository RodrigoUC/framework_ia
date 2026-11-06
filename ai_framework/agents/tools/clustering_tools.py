"""Clustering tools built on :class:`ai_framework.models.unsupervised.Clustering`.

They return metrics and aggregated profiles; per-row labels never reach the LLM.
"""

from __future__ import annotations

from ai_framework.models.unsupervised import Clustering
from ai_framework.utils.results import ResultadoCluster
from ai_framework.utils.unsupervised import (
    ALGORITMOS_CLUSTERING_COMPARACION,
    despachar_algoritmo,
    evaluar_hac,
    evaluar_kmeans,
)
from ai_framework.agents.tools.registry import tool

MAX_K = 20
MIN_PROFILE_ROWS = 3  # the mean of a smaller group almost reveals individual rows
MAX_DISTINCTIVE = 3
HAC_METHODS = ("ward", "average", "complete", "single")


def create_clustering(blackboard, **config) -> Clustering:
    """Stores on the blackboard the model built with the active analysis settings."""
    model = Clustering(dataframe=blackboard.datos, **config)
    blackboard.results["cluster_model"] = model
    return model


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _records(table) -> list[dict]:
    return table.round(4).astype(object).where(table.notna(), None).to_dict("records")


def summarize_result(result: ResultadoCluster) -> dict:
    """Size of each group and the variables that set it apart from the mean, in std units."""
    matrix = result.datos_preparados.matriz
    sizes = result.etiquetas.value_counts().sort_index()
    deviations = matrix.std().replace(0, 1)
    means = (matrix.groupby(result.etiquetas).mean() - matrix.mean()) / deviations
    groups = []
    for group, size in sizes.items():
        description = {
            "cluster": int(group),
            "size": int(size),
            "share": _round(size / len(matrix)),
        }
        if size >= MIN_PROFILE_ROWS:
            distinctive = means.loc[group].sort_values(key=abs, ascending=False)
            description["distinctive"] = {
                str(c): _round(v) for c, v in distinctive.head(MAX_DISTINCTIVE).items()
            }
        groups.append(description)
    return {
        "algorithm": result.algoritmo,
        "n_clusters": int(len(sizes)),
        "silhouette": _round(result.silhouette),
        "inertia": _round(result.inercia),
        "groups": groups,
    }


@tool(
    "Compara inercia y silhouette de K-Means para un intervalo de k.",
    k_min="Menor número de clusters, al menos 2.",
    k_max=f"Mayor número de clusters, hasta {MAX_K}.",
)
def evaluate_k(blackboard, k_min: int = 2, k_max: int = 10) -> dict:
    matrix = blackboard.results["cluster_model"].preparar_matriz().matriz
    table = evaluar_kmeans(matrix, k_min, min(k_max, MAX_K))
    return {"rows": _records(table)}


@tool(
    "Compara silhouette de clustering jerárquico por método y número de clusters.",
    k_min="Menor número de clusters, al menos 2.",
    k_max=f"Mayor número de clusters, hasta {MAX_K}.",
    methods=f"Subconjunto de {HAC_METHODS}; si se omite, todos.",
)
def evaluate_hierarchical(
    blackboard, k_min: int = 2, k_max: int = 10, methods: list[str] | None = None
) -> dict:
    matrix = blackboard.results["cluster_model"].preparar_matriz().matriz
    table = evaluar_hac(matrix, k_min, min(k_max, MAX_K), metodos=tuple(methods or HAC_METHODS))
    return {"rows": _records(table)}


@tool(
    "Agrupa con un algoritmo y describe los grupos con métricas y variables distintivas.",
    algorithm=f"Uno de {ALGORITMOS_CLUSTERING_COMPARACION}.",
    n_clusters="Número de clusters.",
    hac_method=f"Sólo para HAC: uno de {HAC_METHODS}; por defecto ward.",
)
def run_clustering(
    blackboard, algorithm: str, n_clusters: int, hac_method: str | None = None
) -> dict:
    if algorithm not in ALGORITMOS_CLUSTERING_COMPARACION:
        raise ValueError(
            f"Algoritmo no soportado {algorithm!r}; use {ALGORITMOS_CLUSTERING_COMPARACION}."
        )
    options = {"metodo": hac_method or "ward"} if algorithm == "HAC" else {}
    result = despachar_algoritmo(
        blackboard.results["cluster_model"], algorithm, n_clusters=n_clusters, **options
    )
    return summarize_result(result)


CLUSTERING_TOOLS = (evaluate_k, evaluate_hierarchical, run_clustering)
