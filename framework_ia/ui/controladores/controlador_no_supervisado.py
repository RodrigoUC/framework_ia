"""Controlador de casos de uso y muestreo para aprendizaje no supervisado."""

from __future__ import annotations

import pandas as pd

from framework_ia.modelos.no_supervisado import Cluster
from framework_ia.resultados import (
    ResultadoACP,
    ResultadoCluster,
    ResultadoProyeccion,
)
from framework_ia.utils.no_supervisado import (
    evaluar_hac,
    evaluar_kmeans,
)


def _columnas_numericas(datos: pd.DataFrame) -> list[str]:
    """Retorna las variables numéricas del DataFrame activo."""
    return datos.select_dtypes(include="number").columns.tolist()


def _columnas_categoricas(datos: pd.DataFrame) -> list[str]:
    """Retorna las variables categóricas del DataFrame activo."""
    return datos.select_dtypes(exclude="number").columns.tolist()


def _muestrear(datos: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    """Limita análisis costosos de forma reproducible."""
    if len(datos) <= cantidad:
        return datos.copy()
    return datos.sample(n=cantidad, random_state=42).sort_index()


def _crear_reductor(datos: pd.DataFrame, configuracion: dict) -> Cluster:
    """Construye el modelo usado para ACP, t-SNE y UMAP desde la interfaz."""
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def _crear_cluster(datos: pd.DataFrame, configuracion: dict) -> Cluster:
    """Construye un modelo de agrupamiento sin lógica específica de Streamlit."""
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def ejecutar_acp(
    datos: pd.DataFrame, configuracion: dict, n_componentes: int = 2
) -> ResultadoACP:
    """Ejecuta el análisis de componentes principales según la configuración."""
    muestra = _muestrear(datos, configuracion["filas"])
    modelo = _crear_reductor(muestra, configuracion)
    return modelo.ACP(n_componentes=n_componentes)


def ejecutar_kmeans(
    datos: pd.DataFrame, configuracion: dict, clusters: int
) -> ResultadoCluster:
    """Ejecuta el algoritmo K-Means con la muestra configurada."""
    muestra = _muestrear(datos, configuracion["filas"])
    modelo = _crear_cluster(muestra, configuracion)
    return modelo.K_means(clusters)


def evaluar_kmeans_k(
    datos: pd.DataFrame, configuracion: dict, k_min: int = 2, k_max: int = 10
) -> pd.DataFrame:
    """Evalúa valores de k para K-Means mediante inercia y silhouette."""
    muestra = _muestrear(datos, configuracion["filas"])
    modelo = _crear_cluster(muestra, configuracion)
    limite = min(k_max, len(muestra) - 1)
    if k_min > limite:
        raise ValueError("No hay suficientes filas para evaluar el rango de clusters.")
    matriz = modelo.preparar_matriz().matriz
    return evaluar_kmeans(matriz, k_min=k_min, k_max=limite)


def ejecutar_kmedoids(
    datos: pd.DataFrame, configuracion: dict, clusters: int
) -> ResultadoCluster:
    """Ejecuta el algoritmo K-Medoids con la muestra configurada."""
    muestra = _muestrear(datos, configuracion["filas"])
    modelo = _crear_cluster(muestra, configuracion)
    return modelo.K_medoids(clusters)


def ejecutar_hac(
    datos: pd.DataFrame,
    configuracion: dict,
    clusters: int,
    metodo: str = "ward",
    filas_hac: int = 800,
) -> ResultadoCluster:
    """Ejecuta clustering jerárquico aglomerativo sobre la muestra acotada."""
    muestra = _muestrear(datos, filas_hac)
    modelo = _crear_cluster(muestra, configuracion)
    return modelo.HAC(clusters, metodo=metodo)


def comparar_metodos_hac(
    datos: pd.DataFrame,
    configuracion: dict,
    filas_hac: int = 800,
    k_min: int = 2,
    k_max: int = 8,
) -> pd.DataFrame:
    """Compara los métodos de vinculación de HAC para un rango de clusters."""
    muestra = _muestrear(datos, filas_hac)
    modelo = _crear_cluster(muestra, configuracion)
    limite = min(k_max, len(muestra) - 1)
    if k_min > limite:
        raise ValueError(
            "No hay suficientes filas para evaluar el rango de clusters en HAC."
        )
    matriz = modelo.preparar_matriz().matriz
    return evaluar_hac(matriz, k_min=k_min, k_max=limite)


def ejecutar_tsne(
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
        else _muestrear(datos, filas)
    )
    modelo = _crear_reductor(muestra, configuracion)
    return modelo.TSNE(perplexity=perplexity, max_iter=iteraciones)


def ejecutar_umap(
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
        else _muestrear(datos, filas)
    )
    modelo = _crear_reductor(muestra, configuracion)
    return modelo.UMAP(n_neighbors=vecinos, min_dist=distancia)
