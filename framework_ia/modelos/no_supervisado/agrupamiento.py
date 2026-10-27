"""Algoritmos de agrupamiento (clustering): K-Means, K-Medoids y HAC."""

from __future__ import annotations

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

from .no_supervisados import NoSupervisado
from ...resultados import ResultadoCluster
from ...utils.evaluacion_cluster import (
    calcular_centroides,
    construir_resultado_cluster,
    evaluar_hac,
    evaluar_kmeans,
    validar_numero_clusters,
)
from ...utils.k_medoids import ModeloKMedoids


class Cluster(NoSupervisado):
    """Ejecuta algoritmos de agrupamiento: K-Means, K-Medoids y HAC."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.centroides: pd.DataFrame | None = None
        self.inercia: float | None = None

    def K_means(
        self,
        n_clusters: int = 3,
        *,
        random_state: int = 42,
        n_init: int = 20,
        max_iter: int = 500,
    ) -> ResultadoCluster:
        """Agrupa observaciones mediante centroides K-Means."""
        preparados = self.preparar_matriz()
        validar_numero_clusters(preparados.matriz, n_clusters)
        modelo = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        etiquetas = modelo.fit_predict(preparados.matriz)
        centroides = pd.DataFrame(
            modelo.cluster_centers_,
            columns=preparados.matriz.columns,
            index=pd.Index(range(n_clusters), name="cluster"),
        )
        resultado = self._construir_resultado(
            algoritmo="K-Means",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo=modelo,
            inercia=float(modelo.inertia_),
        )
        self.inercia = resultado.inercia
        return resultado

    def K_medoids(
        self,
        n_clusters: int = 3,
        *,
        random_state: int = 42,
        max_iter: int = 500,
        metric: str = "manhattan",
    ) -> ResultadoCluster:
        """Agrupa observaciones alrededor de medoides mediante distancias."""
        preparados = self.preparar_matriz()
        validar_numero_clusters(preparados.matriz, n_clusters)
        modelo = ModeloKMedoids(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            metric=metric,
        )
        etiquetas = modelo.fit_predict(preparados.matriz)
        centroides = pd.DataFrame(
            modelo.cluster_centers_,
            columns=preparados.matriz.columns,
            index=pd.Index(range(n_clusters), name="cluster"),
        )
        return self._construir_resultado(
            algoritmo="K-Medoids",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo=modelo,
            inercia=float(modelo.inertia_),
        )

    def HAC(
        self,
        n_clusters: int = 3,
        *,
        metodo: str = "ward",
        metrica: str = "euclidean",
    ) -> ResultadoCluster:
        """Ejecuta clustering jerárquico aglomerativo conservando dendrograma."""
        metodos_validos = {"ward", "complete", "average", "single"}
        if metodo not in metodos_validos:
            raise ValueError(f"Método jerárquico no soportado: {metodo}")
        if metodo == "ward" and metrica != "euclidean":
            raise ValueError("El método ward únicamente admite distancia euclidean.")

        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        validar_numero_clusters(matriz, n_clusters)
        matriz_vinculacion = linkage(
            matriz.to_numpy(), method=metodo, metric=metrica
        )
        etiquetas = (
            fcluster(matriz_vinculacion, t=n_clusters, criterion="maxclust") - 1
        )
        centroides = calcular_centroides(matriz, etiquetas)
        return self._construir_resultado(
            algoritmo=f"HAC-{metodo}",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo={"metodo": metodo, "metrica": metrica},
            matriz_vinculacion=matriz_vinculacion,
        )

    def ajustar(self, algoritmo: str = "KMEANS", **kwargs) -> ResultadoCluster:
        """Despacha el algoritmo solicitado a través de una interfaz común."""
        metodos = {
            "KMEANS": self.K_means,
            "KMEDOIDS": self.K_medoids,
            "HAC": self.HAC,
        }
        clave = algoritmo.upper().replace("-", "").replace("_", "")
        if clave not in metodos:
            raise ValueError(f"Algoritmo no soportado: {algoritmo}")
        return metodos[clave](**kwargs)

    def evaluar_kmeans(
        self,
        k_min: int = 2,
        k_max: int = 10,
        *,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compara inercia y silhouette para un intervalo de valores de k."""
        return evaluar_kmeans(
            self.preparar_matriz().matriz,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
        )

    def evaluar_hac(
        self,
        k_min: int = 2,
        k_max: int = 10,
        *,
        metodos: tuple[str, ...] = ("ward", "average", "complete", "single"),
    ) -> pd.DataFrame:
        """Compara silhouette para métodos jerárquicos y cantidades de clusters."""
        return evaluar_hac(
            self.preparar_matriz().matriz,
            k_min=k_min,
            k_max=k_max,
            metodos=metodos,
        )

    def _construir_resultado(
        self,
        *,
        algoritmo: str,
        preparados,
        etiquetas,
        centroides: pd.DataFrame,
        modelo,
        inercia: float | None = None,
        matriz_vinculacion=None,
    ) -> ResultadoCluster:
        """Construye el resultado de agrupamiento y actualiza el estado interno."""
        resultado = construir_resultado_cluster(
            algoritmo=algoritmo,
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo=modelo,
            inercia=inercia,
            matriz_vinculacion=matriz_vinculacion,
            proyeccion_2d=self.proyectar_2d(preparados.matriz),
        )
        self.modelo = modelo
        self.etiquetas = resultado.etiquetas
        self.centroides = centroides
        self._registrar_resultado(algoritmo, resultado.etiquetas.nunique(), resultado.silhouette)
        return resultado
