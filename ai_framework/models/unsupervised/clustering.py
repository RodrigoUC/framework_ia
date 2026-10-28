"""Algoritmos de agrupamiento y proyecciones no supervisadas: K-Means, K-Medoids, HAC, ACP, t-SNE y UMAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .unsupervised import OptionalDependencyError, Unsupervised
from ai_framework.utils.results import (
    ResultadoACP,
    ResultadoCluster,
    ResultadoProyeccion,
)
from ai_framework.utils import (
    configurar_cache_numba,
    crear_tsne,
)
from ai_framework.utils.unsupervised.cluster_evaluation import (
    calcular_centroides,
    construir_resultado_cluster,
    validar_numero_clusters,
)
from ai_framework.utils.unsupervised.k_medoids import ModeloKMedoids


class Clustering(Unsupervised):
    """Ejecuta algoritmos de agrupamiento (K-Means, K-Medoids, HAC) y proyecciones (ACP, t-SNE, UMAP)."""

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

    def ACP(self, n_componentes: int = 2) -> ResultadoACP:
        """Calcula componentes principales, cargas y varianza explicada."""
        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        maximo = min(matriz.shape)
        if not 1 <= n_componentes <= maximo:
            raise ValueError(f"n_componentes debe estar entre 1 y {maximo}.")

        modelo = PCA(n_components=n_componentes)
        valores = modelo.fit_transform(matriz)
        nombres = [f"CP{i + 1}" for i in range(n_componentes)]
        coordenadas = pd.DataFrame(valores, index=matriz.index, columns=nombres)

        cargas_np = modelo.components_.T * np.sqrt(modelo.explained_variance_)
        cargas = pd.DataFrame(cargas_np, index=matriz.columns, columns=nombres)
        varianza = pd.Series(
            modelo.explained_variance_ratio_ * 100,
            index=nombres,
            name="varianza_explicada_porcentaje",
        )
        acumulada = varianza.cumsum().rename("varianza_acumulada_porcentaje")
        self.modelo = modelo
        return ResultadoACP(
            coordenadas=coordenadas,
            cargas=cargas,
            varianza_explicada=varianza,
            varianza_acumulada=acumulada,
            datos_preparados=preparados,
            modelo=modelo,
        )

    def TSNE(
        self,
        *,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Genera una proyección t-SNE bidimensional."""
        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        if len(matriz) < 3:
            raise ValueError("t-SNE requiere al menos tres filas.")
        if not 1 <= perplexity < len(matriz):
            raise ValueError(
                f"perplexity debe ser menor que el número de filas ({len(matriz)})."
            )

        modelo = crear_tsne(
            perplexity=perplexity, max_iter=max_iter, random_state=random_state
        )
        valores = modelo.fit_transform(matriz)
        coordenadas = pd.DataFrame(
            valores,
            index=matriz.index,
            columns=["TSNE1", "TSNE2"],
        )
        self.modelo = modelo
        return ResultadoProyeccion(
            algoritmo="t-SNE",
            coordenadas=coordenadas,
            datos_preparados=preparados,
            modelo=modelo,
        )

    def UMAP(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Genera una proyección UMAP bidimensional reproducible."""
        configurar_cache_numba()
        try:
            import umap.umap_ as umap
        except (ImportError, RuntimeError) as exc:
            raise OptionalDependencyError(
                "No fue posible inicializar UMAP. Instale 'umap-learn' y "
                "verifique que Numba tenga un directorio de caché escribible. "
                f"Detalle técnico: {exc}"
            ) from exc

        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        if len(matriz) < 3:
            raise ValueError("UMAP requiere al menos tres filas.")
        if not 2 <= n_neighbors < len(matriz):
            raise ValueError(
                f"n_neighbors debe estar entre 2 y {len(matriz) - 1}."
            )

        modelo = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=1,
        )
        valores = modelo.fit_transform(matriz)
        coordenadas = pd.DataFrame(
            valores,
            index=matriz.index,
            columns=["UMAP1", "UMAP2"],
        )
        self.modelo = modelo
        return ResultadoProyeccion(
            algoritmo="UMAP",
            coordenadas=coordenadas,
            datos_preparados=preparados,
            modelo=modelo,
        )


def __getattr__(name: str):
    if name == "Cluster":
        import ai_framework.utils.aliases as aliases
        return aliases.Cluster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
