"""Algoritmos no supervisados del framework: agrupamiento, ACP, t-SNE y UMAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .algoritmos_cluster import ModeloKMedoids
from .base import NoSupervisado
from ...resultados import (
    DatosPreparados,
    ResultadoACP,
    ResultadoCluster,
    ResultadoProyeccion,
)
from ...utils import configurar_cache_numba, crear_tsne


class DependenciaOpcionalError(ImportError):
    """Indica que una capacidad opcional requiere instalar una dependencia."""


class Cluster(NoSupervisado):
    """Ejecuta K-Means, K-Medoids, HAC y las proyecciones ACP, t-SNE y UMAP."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.centroides: pd.DataFrame | None = None
        self.inercia: float | None = None

    def ajustar(self, algoritmo: str = "KMEANS", **kwargs):
        """Despacha el algoritmo solicitado con una interfaz común."""
        metodos = {
            "KMEANS": self.K_means,
            "K-MEANS": self.K_means,
            "KMEDOIDS": self.K_medoids,
            "K-MEDOIDS": self.K_medoids,
            "HAC": self.HAC,
            "HPC": self.HAC,
            "ACP": self.ACP,
            "TSNE": self.TSNE,
            "T-SNE": self.TSNE,
            "UMAP": self.UMAP,
        }
        try:
            metodo = metodos[algoritmo.upper()]
        except KeyError as exc:
            raise ValueError(f"Algoritmo no soportado: {algoritmo}") from exc
        return metodo(**kwargs)

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
        self._validar_numero_clusters(preparados.matriz, n_clusters)
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
        """Agrupa observaciones alrededor de medoides mediante distancia."""
        preparados = self.preparar_matriz()
        self._validar_numero_clusters(preparados.matriz, n_clusters)
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

    def k_means(self, *args, **kwargs) -> ResultadoCluster:
        """Alias pythonico de :meth:`K_means`."""
        return self.K_means(*args, **kwargs)

    def k_medoids(self, *args, **kwargs) -> ResultadoCluster:
        """Alias pythonico de :meth:`K_medoids`."""
        return self.K_medoids(*args, **kwargs)

    def HAC(
        self,
        n_clusters: int = 3,
        *,
        metodo: str = "ward",
        metrica: str = "euclidean",
    ) -> ResultadoCluster:
        """Ejecuta clustering jerárquico aglomerativo y conserva el dendrograma."""
        metodos_validos = {"ward", "complete", "average", "single"}
        if metodo not in metodos_validos:
            raise ValueError(f"Método jerárquico no soportado: {metodo}")
        if metodo == "ward" and metrica != "euclidean":
            raise ValueError("El método ward únicamente admite distancia euclidean.")

        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        self._validar_numero_clusters(matriz, n_clusters)
        matriz_vinculacion = linkage(
            matriz.to_numpy(), method=metodo, metric=metrica
        )
        etiquetas = fcluster(
            matriz_vinculacion, t=n_clusters, criterion="maxclust"
        ) - 1
        centroides = self._calcular_centroides(matriz, etiquetas)
        return self._construir_resultado(
            algoritmo=f"HAC-{metodo}",
            preparados=preparados,
            etiquetas=etiquetas,
            centroides=centroides,
            modelo={"metodo": metodo, "metrica": metrica},
            matriz_vinculacion=matriz_vinculacion,
        )

    def HPC(self, *args, **kwargs) -> ResultadoCluster:
        """Alias semántico de HAC conservado por compatibilidad del framework."""
        return self.HAC(*args, **kwargs)

    def hac(self, *args, **kwargs) -> ResultadoCluster:
        """Alias pythonico de :meth:`HAC`."""
        return self.HAC(*args, **kwargs)

    # ==========================================================
    # Reducción dimensional: ACP, t-SNE y UMAP
    # ==========================================================

    def ACP(self, n_componentes: int = 2) -> ResultadoACP:
        """Calcula componentes, cargas y varianza explicada con sklearn."""
        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        maximo = min(matriz.shape)
        if not 1 <= n_componentes <= maximo:
            raise ValueError(
                f"n_componentes debe estar entre 1 y {maximo}."
            )

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

    def acp(self, n_componentes: int = 2) -> ResultadoACP:
        """Alias pythonico de :meth:`ACP`."""
        return self.ACP(n_componentes=n_componentes)

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

    def tsne(self, *args, **kwargs) -> ResultadoProyeccion:
        """Alias pythonico de :meth:`TSNE`."""
        return self.TSNE(*args, **kwargs)

    def UMAP(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Genera una proyección UMAP bidimensional y reproducible.

        ``umap-learn`` usa Numba durante la importación; si el directorio de
        caché no es escribible se configura uno temporal (ver
        :func:`utils.configurar_cache_numba`).
        """
        configurar_cache_numba()
        try:
            import umap.umap_ as umap
        except (ImportError, RuntimeError) as exc:
            raise DependenciaOpcionalError(
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

    def umap(self, *args, **kwargs) -> ResultadoProyeccion:
        """Alias pythonico de :meth:`UMAP`."""
        return self.UMAP(*args, **kwargs)

    def evaluar_kmeans(
        self,
        k_min: int = 2,
        k_max: int = 10,
        *,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compara inercia y silhouette para un intervalo de valores de k."""
        matriz = self.preparar_matriz().matriz
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
        self,
        k_min: int = 2,
        k_max: int = 10,
        *,
        metodos: tuple[str, ...] = ("ward", "average", "complete", "single"),
    ) -> pd.DataFrame:
        """Compara silhouette para métodos y cantidades de clusters."""
        matriz = self.preparar_matriz().matriz
        limite = min(k_max, len(matriz) - 1)
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
                filas.append(
                    {"metodo": metodo, "k": k, "silhouette": valor}
                )
        return pd.DataFrame(filas)

    def _construir_resultado(
        self,
        *,
        algoritmo: str,
        preparados: DatosPreparados,
        etiquetas,
        centroides: pd.DataFrame,
        modelo,
        inercia: float | None = None,
        matriz_vinculacion=None,
    ) -> ResultadoCluster:
        """Normaliza la salida y actualiza el estado de la instancia."""
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
        resultado = ResultadoCluster(
            algoritmo=algoritmo,
            etiquetas=serie_etiquetas,
            centroides=centroides,
            proyeccion_2d=self._proyectar_2d(preparados.matriz),
            silhouette=silhouette,
            datos_preparados=preparados,
            modelo=modelo,
            inercia=inercia,
            matriz_vinculacion=matriz_vinculacion,
        )
        self.modelo = modelo
        self.etiquetas = serie_etiquetas
        self.centroides = centroides
        self._registrar_resultado(algoritmo, cantidad, silhouette)
        return resultado

    @staticmethod
    def _validar_numero_clusters(matriz: pd.DataFrame, n_clusters: int) -> None:
        """Comprueba que silhouette y el modelo tengan suficientes filas."""
        if not 2 <= n_clusters < len(matriz):
            raise ValueError(
                "n_clusters debe ser al menos 2 y menor que el número de filas."
            )

    @staticmethod
    def _calcular_centroides(matriz: pd.DataFrame, etiquetas) -> pd.DataFrame:
        """Calcula perfiles promedio por cluster en el espacio transformado."""
        temporal = matriz.copy()
        temporal["cluster"] = etiquetas
        return temporal.groupby("cluster", sort=True).mean()

    @staticmethod
    def _proyectar_2d(matriz: pd.DataFrame) -> pd.DataFrame:
        """Proyecta la matriz para visualizar etiquetas sin alterar el modelo."""
        if matriz.shape[1] == 1:
            valores = np.column_stack(
                [matriz.iloc[:, 0].to_numpy(), np.zeros(len(matriz))]
            )
        else:
            valores = PCA(n_components=2).fit_transform(matriz)
        return pd.DataFrame(
            valores,
            index=matriz.index,
            columns=["Componente 1", "Componente 2"],
        )
