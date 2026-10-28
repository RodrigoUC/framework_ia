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
        resultado = construir_resultado_cluster(
            algoritmo="K-Means",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo=modelo,
            inercia=float(modelo.inertia_),
            proyeccion_2d=self.proyectar_2d(preparados.matriz),
        )
        self.modelo = modelo
        self.etiquetas = resultado.etiquetas
        self.centroides = centroides
        self.inercia = resultado.inercia
        self._registrar_resultado(
            "K-Means", resultado.etiquetas.nunique(), resultado.silhouette
        )
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
        resultado = construir_resultado_cluster(
            algoritmo="K-Medoids",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo=modelo,
            inercia=float(modelo.inertia_),
            proyeccion_2d=self.proyectar_2d(preparados.matriz),
        )
        self.modelo = modelo
        self.etiquetas = resultado.etiquetas
        self.centroides = centroides
        self.inercia = resultado.inercia
        self._registrar_resultado(
            "K-Medoids", resultado.etiquetas.nunique(), resultado.silhouette
        )
        return resultado

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
        resultado = construir_resultado_cluster(
            algoritmo=f"HAC-{metodo}",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo={"metodo": metodo, "metrica": metrica},
            matriz_vinculacion=matriz_vinculacion,
            proyeccion_2d=self.proyectar_2d(preparados.matriz),
        )
        self.modelo = {"metodo": metodo, "metrica": metrica}
        self.etiquetas = resultado.etiquetas
        self.centroides = centroides
        self.inercia = None
        self._registrar_resultado(
            f"HAC-{metodo}", resultado.etiquetas.nunique(), resultado.silhouette
        )
        return resultado

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

    # Alias pythonicos
    k_means = K_means
    k_medoids = K_medoids
    hac = HAC
    fit = ajustar
