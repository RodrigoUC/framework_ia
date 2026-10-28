"""Funciones de evaluación, validación y métricas para algoritmos de agrupamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ...resultados import DatosPreparados, ResultadoCluster


def validar_numero_clusters(matriz: pd.DataFrame, n_clusters: int) -> None:
    """Comprueba que el número de clusters sea válido para el tamaño de la matriz."""
    if not 2 <= n_clusters < len(matriz):
        raise ValueError(
            "n_clusters debe ser al menos 2 y menor que el número de filas."
        )


def calcular_centroides(matriz: pd.DataFrame, etiquetas) -> pd.DataFrame:
    """Calcula los perfiles promedio por cluster en el espacio de características."""
    temporal = matriz.copy()
    temporal["cluster"] = etiquetas
    return temporal.groupby("cluster", sort=True).mean()


def evaluar_kmeans(
    matriz: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara inercia y silhouette para un intervalo de valores de k en K-Means."""
    limite = min(k_max, len(matriz) - 1)
    if k_min > limite:
        raise ValueError("No hay suficientes filas para evaluar ese intervalo.")

    filas = []
    for k in range(k_min, limite + 1):
        modelo = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
            max_iter=500,
        )
        etiquetas = modelo.fit_predict(matriz)
        filas.append(
            {
                "k": k,
                "inercia": float(modelo.inertia_),
                "silhouette": float(silhouette_score(matriz, etiquetas)),
            }
        )
    return pd.DataFrame(filas)


def evaluar_hac(
    matriz: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    *,
    metodos: tuple[str, ...] = ("ward", "average", "complete", "single"),
) -> pd.DataFrame:
    """Compara coeficiente de silhouette para métodos jerárquicos y cantidades de k."""
    limite = min(k_max, len(matriz) - 1)
    if k_min > limite:
        raise ValueError("No hay suficientes filas para evaluar el rango de clusters en HAC.")
    filas = []
    for metodo in metodos:
        vinculacion = linkage(matriz.to_numpy(), method=metodo, metric="euclidean")
        for k in range(k_min, limite + 1):
            etiquetas = fcluster(vinculacion, t=k, criterion="maxclust") - 1
            cantidad_real = len(np.unique(etiquetas))
            if 1 < cantidad_real < len(matriz):
                valor = float(silhouette_score(matriz, etiquetas))
            else:
                valor = np.nan
            filas.append({"metodo": metodo, "k": k, "silhouette": valor})
    return pd.DataFrame(filas)


def construir_resultado_cluster(
    *,
    algoritmo: str,
    preparados: DatosPreparados,
    etiquetas,
    centroides: pd.DataFrame,
    modelo,
    inercia: float | None = None,
    matriz_vinculacion=None,
    proyeccion_2d: pd.DataFrame | None = None,
) -> ResultadoCluster:
    """Construye un ResultadoCluster normalizado calculando silhouette si aplica."""
    serie_etiquetas = pd.Series(
        etiquetas,
        index=preparados.matriz.index,
        name="cluster",
        dtype=int,
    )
    cantidad = serie_etiquetas.nunique()
    silhouette = (
        float(silhouette_score(preparados.matriz, serie_etiquetas))
        if 1 < cantidad < len(serie_etiquetas)
        else None
    )
    return ResultadoCluster(
        algoritmo=algoritmo,
        etiquetas=serie_etiquetas,
        centroides=centroides,
        proyeccion_2d=proyeccion_2d,
        silhouette=silhouette,
        datos_preparados=preparados,
        modelo=modelo,
        inercia=inercia,
        matriz_vinculacion=matriz_vinculacion,
    )
