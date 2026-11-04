"""Controller for use cases and sampling in unsupervised learning."""

from __future__ import annotations

import pandas as pd

from ai_framework.models.unsupervised import Cluster
from ai_framework.utils.results import (
    ResultadoACP,
    ResultadoCluster,
    ResultadoProyeccion,
)
from ai_framework.utils.unsupervised import (
    evaluar_hac,
    evaluar_kmeans,
)


def _numeric_columns(datos: pd.DataFrame) -> list[str]:
    """Retorna las variables numéricas del DataFrame activo."""
    return datos.select_dtypes(include="number").columns.tolist()


def _categorical_columns(datos: pd.DataFrame) -> list[str]:
    """Retorna las variables categóricas del DataFrame activo."""
    return datos.select_dtypes(exclude="number").columns.tolist()


def _sample_data(datos: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    """Limita análisis costosos de forma reproducible."""
    if len(datos) <= cantidad:
        return datos.copy()
    return datos.sample(n=cantidad, random_state=42).sort_index()


def _create_reducer(datos: pd.DataFrame, configuracion: dict) -> Cluster:
    """Construye el modelo usado para ACP, t-SNE y UMAP desde la interfaz."""
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def _create_cluster(datos: pd.DataFrame, configuracion: dict) -> Cluster:
    """Construye un modelo de agrupamiento sin lógica específica de Streamlit."""
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def run_pca(
    datos: pd.DataFrame, configuracion: dict, n_componentes: int = 2
) -> ResultadoACP:
    """Execute principal component analysis according to configuration."""
    muestra = _sample_data(datos, configuracion["filas"])
    modelo = _create_reducer(muestra, configuracion)
    return modelo.ACP(n_componentes=n_componentes)


def run_kmeans(
    datos: pd.DataFrame, configuracion: dict, clusters: int
) -> ResultadoCluster:
    """Execute K-Means algorithm with the configured sample."""
    muestra = _sample_data(datos, configuracion["filas"])
    modelo = _create_cluster(muestra, configuracion)
    return modelo.K_means(clusters)


def evaluate_kmeans_k(
    datos: pd.DataFrame, configuracion: dict, k_min: int = 2, k_max: int = 10
) -> pd.DataFrame:
    """Evalúa valores de k para K-Means mediante inercia y silhouette."""
    muestra = _sample_data(datos, configuracion["filas"])
    modelo = _create_cluster(muestra, configuracion)
    limite = min(k_max, len(muestra) - 1)
    if k_min > limite:
        raise ValueError("No hay suficientes filas para evaluar el rango de clusters.")
    matriz = modelo.preparar_matriz().matriz
    return evaluar_kmeans(matriz, k_min=k_min, k_max=limite)


def run_kmedoids(
    datos: pd.DataFrame, configuracion: dict, clusters: int
) -> ResultadoCluster:
    """Execute K-Medoids algorithm with the configured sample."""
    muestra = _sample_data(datos, configuracion["filas"])
    modelo = _create_cluster(muestra, configuracion)
    return modelo.K_medoids(clusters)


def run_hac(
    datos: pd.DataFrame,
    configuracion: dict,
    clusters: int,
    metodo: str = "ward",
    filas_hac: int = 800,
) -> ResultadoCluster:
    """Ejecuta clustering jerárquico aglomerativo sobre la muestra acotada."""
    muestra = _sample_data(datos, filas_hac)
    modelo = _create_cluster(muestra, configuracion)
    return modelo.HAC(clusters, metodo=metodo)


def compare_hac_methods(
    datos: pd.DataFrame,
    configuracion: dict,
    filas_hac: int = 800,
    k_min: int = 2,
    k_max: int = 8,
) -> pd.DataFrame:
    """Compare HAC linkage methods across a range of cluster counts."""
    muestra = _sample_data(datos, filas_hac)
    modelo = _create_cluster(muestra, configuracion)
    limite = min(k_max, len(muestra) - 1)
    if k_min > limite:
        raise ValueError(
            "No hay suficientes filas para evaluar el rango de clusters en HAC."
        )
    matriz = modelo.preparar_matriz().matriz
    return evaluar_hac(matriz, k_min=k_min, k_max=limite)


def run_tsne(
    datos: pd.DataFrame,
    configuracion: dict,
    filas: int = 2000,
    perplexity: float = 30.0,
    iteraciones: int = 1000,
    etiquetas=None,
) -> ResultadoProyeccion:
    """Ejecuta t-SNE sobre el conjunto de variables seleccionadas."""
    muestra = (
        datos.loc[etiquetas.index[:filas]]
        if etiquetas is not None
        else _sample_data(datos, filas)
    )
    modelo = _create_reducer(muestra, configuracion)
    return modelo.TSNE(perplexity=perplexity, max_iter=iteraciones)


def run_umap(
    datos: pd.DataFrame,
    configuracion: dict,
    filas: int = 3000,
    vecinos: int = 15,
    distancia: float = 0.1,
    etiquetas=None,
) -> ResultadoProyeccion:
    """Ejecuta UMAP sobre el conjunto de variables seleccionadas."""
    muestra = (
        datos.loc[etiquetas.index[:filas]]
        if etiquetas is not None
        else _sample_data(datos, filas)
    )
    modelo = _create_reducer(muestra, configuracion)
    return modelo.UMAP(n_neighbors=vecinos, min_dist=distancia)

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
