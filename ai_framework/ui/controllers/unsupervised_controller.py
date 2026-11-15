"""Controller for use cases and sampling in unsupervised learning."""

from __future__ import annotations

import pandas as pd

from ai_framework.models.unsupervised import Cluster
from ai_framework.utils.results import (
    PCAResult,
    ClusterResult,
    ProjectionResult,
    ResultadoACP,
    ResultadoCluster,
    ResultadoProyeccion,
)
from ai_framework.utils.unsupervised import (
    evaluate_hac,
    evaluate_kmeans,
    evaluar_hac,
    evaluar_kmeans,
)


def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric variables of active DataFrame."""
    return data.select_dtypes(include="number").columns.tolist()


def _categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return categorical variables of active DataFrame."""
    return data.select_dtypes(exclude="number").columns.tolist()


def _sample_data(data: pd.DataFrame, count: int) -> pd.DataFrame:
    """Limit expensive analysis reproducibly."""
    if len(data) <= count:
        return data.copy()
    return data.sample(n=count, random_state=42).sort_index()


def _create_reducer(data: pd.DataFrame, config: dict) -> Cluster:
    """Build model used for PCA, t-SNE, and UMAP from UI."""
    return Cluster(
        dataframe=data,
        features=config["features"],
        incluir_categoricas=config["incluir_categoricas"],
        estandarizar=config["estandarizar"],
    )


def _create_cluster(data: pd.DataFrame, config: dict) -> Cluster:
    """Build clustering model without Streamlit-specific logic."""
    return Cluster(
        dataframe=data,
        features=config["features"],
        incluir_categoricas=config["incluir_categoricas"],
        estandarizar=config["estandarizar"],
    )


def run_pca(
    data: pd.DataFrame, config: dict, n_components: int = 2, **kwargs
) -> PCAResult:
    """Execute principal component analysis according to configuration."""
    n_comp = kwargs.get("n_componentes", n_components)
    sample = _sample_data(data, config["filas"])
    model = _create_reducer(sample, config)
    return model.ACP(n_componentes=n_comp)


def run_kmeans(
    data: pd.DataFrame, config: dict, clusters: int
) -> ClusterResult:
    """Execute K-Means algorithm with the configured sample."""
    sample = _sample_data(data, config["filas"])
    model = _create_cluster(sample, config)
    return model.K_means(clusters)


def evaluate_kmeans_k(
    data: pd.DataFrame, config: dict, k_min: int = 2, k_max: int = 10
) -> pd.DataFrame:
    """Evaluate k values for K-Means using inertia and silhouette."""
    sample = _sample_data(data, config["filas"])
    model = _create_cluster(sample, config)
    limit = min(k_max, len(sample) - 1)
    if k_min > limit:
        raise ValueError("Not enough rows to evaluate cluster range.")
    matrix = model.preparar_matriz().matriz
    return evaluate_kmeans(matrix, k_min=k_min, k_max=limit)


def run_kmedoids(
    data: pd.DataFrame, config: dict, clusters: int
) -> ClusterResult:
    """Execute K-Medoids algorithm with the configured sample."""
    sample = _sample_data(data, config["filas"])
    model = _create_cluster(sample, config)
    return model.K_medoids(clusters)


def run_hac(
    data: pd.DataFrame,
    config: dict,
    clusters: int,
    method: str = "ward",
    rows_hac: int = 800,
    **kwargs,
) -> ClusterResult:
    """Execute hierarchical agglomerative clustering on bounded sample."""
    metodo = kwargs.get("metodo", method)
    filas = kwargs.get("filas_hac", rows_hac)
    sample = _sample_data(data, filas)
    model = _create_cluster(sample, config)
    return model.HAC(clusters, metodo=metodo)


def compare_hac_methods(
    data: pd.DataFrame,
    config: dict,
    rows_hac: int = 800,
    k_min: int = 2,
    k_max: int = 8,
    **kwargs,
) -> pd.DataFrame:
    """Compare HAC linkage methods across a range of cluster counts."""
    filas = kwargs.get("filas_hac", rows_hac)
    sample = _sample_data(data, filas)
    model = _create_cluster(sample, config)
    limit = min(k_max, len(sample) - 1)
    if k_min > limit:
        raise ValueError(
            "Not enough rows to evaluate cluster range in HAC."
        )
    matrix = model.preparar_matriz().matriz
    return evaluate_hac(matrix, k_min=k_min, k_max=limit)


def run_tsne(
    data: pd.DataFrame,
    config: dict,
    rows: int = 2000,
    perplexity: float = 30.0,
    iterations: int = 1000,
    labels=None,
    **kwargs,
) -> ProjectionResult:
    """Execute t-SNE over selected set of variables."""
    filas = kwargs.get("filas", rows)
    iteraciones = kwargs.get("iteraciones", iterations)
    etiquetas = kwargs.get("etiquetas", labels)
    sample = (
        data.loc[etiquetas.index[:filas]]
        if etiquetas is not None
        else _sample_data(data, filas)
    )
    model = _create_reducer(sample, config)
    return model.TSNE(perplexity=perplexity, max_iter=iteraciones)


def run_umap(
    data: pd.DataFrame,
    config: dict,
    rows: int = 3000,
    neighbors: int = 15,
    distance: float = 0.1,
    labels=None,
    **kwargs,
) -> ProjectionResult:
    """Execute UMAP over selected set of variables."""
    filas = kwargs.get("filas", rows)
    vecinos = kwargs.get("vecinos", neighbors)
    distancia = kwargs.get("distancia", distance)
    etiquetas = kwargs.get("etiquetas", labels)
    sample = (
        data.loc[etiquetas.index[:filas]]
        if etiquetas is not None
        else _sample_data(data, filas)
    )
    model = _create_reducer(sample, config)
    return model.UMAP(n_neighbors=vecinos, min_dist=distancia)

# Backward compatibility aliases
ejecutar_acp = run_pca
ejecutar_kmeans = run_kmeans
evaluar_kmeans_k = evaluate_kmeans_k
ejecutar_kmedoids = run_kmedoids
ejecutar_hac = run_hac
comparar_metodos_hac = compare_hac_methods
ejecutar_tsne = run_tsne
ejecutar_umap = run_umap

# Backward compatibility aliases
_columnas_numericas = _numeric_columns
_columnas_categoricas = _categorical_columns
_muestrear = _sample_data
_crear_reductor = _create_reducer
_crear_cluster = _create_cluster
