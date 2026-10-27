"""Abstracciones base y métodos comunes compartidos por análisis no supervisados."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ...datos.eda import EDA
from ...datos.preprocesamiento import PreprocesadorNoSupervisado
from ...resultados import (
    DatosPreparados,
    ResultadoACP,
    ResultadoProyeccion,
)
from ...utils import (
    ConfiguracionPreprocesamiento,
    configurar_cache_numba,
    crear_tsne,
)


class DependenciaOpcionalError(ImportError):
    """Indica que una capacidad opcional requiere instalar una dependencia."""


class NoSupervisado(EDA, ABC):
    """Clase base para análisis no supervisados: preprocesamiento, proyecciones y benchmark."""

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        ruta_datos=None,
        *,
        features=None,
        incluir_categoricas: bool = False,
        estandarizar: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dataframe=dataframe, ruta_datos=ruta_datos, **kwargs)
        columnas = tuple(features) if features is not None else None
        self.configuracion = ConfiguracionPreprocesamiento(
            columnas=columnas,
            incluir_categoricas=incluir_categoricas,
            estandarizar=estandarizar,
        )
        self.modelo = None
        self.etiquetas = None
        self._historial: list[dict[str, int | float | str]] = []
        self._datos_preparados: DatosPreparados | None = None

    def preparar_matriz(self, *, refrescar: bool = False) -> DatosPreparados:
        """Obtiene una matriz numérica reutilizable por los análisis no supervisados."""
        if self._datos_preparados is None or refrescar:
            preprocesador = PreprocesadorNoSupervisado(self.configuracion)
            self._datos_preparados = preprocesador.ajustar_transformar(self.datos)
        return self._datos_preparados

    def proyectar_2d(self, matriz: pd.DataFrame | None = None) -> pd.DataFrame:
        """Proyecta observaciones en 2 dimensiones para análisis visual."""
        matriz_trabajo = (
            matriz if matriz is not None else self.preparar_matriz().matriz
        )
        if matriz_trabajo.shape[1] == 1:
            valores = np.column_stack(
                [matriz_trabajo.iloc[:, 0].to_numpy(), np.zeros(len(matriz_trabajo))]
            )
        else:
            valores = PCA(n_components=2).fit_transform(matriz_trabajo)
        return pd.DataFrame(
            valores,
            index=matriz_trabajo.index,
            columns=["Componente 1", "Componente 2"],
        )

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

    def benchmark(self) -> pd.DataFrame:
        """Retorna las métricas registradas durante la instancia actual."""
        return pd.DataFrame(self._historial)

    def _registrar_resultado(
        self,
        algoritmo: str,
        n_clusters: int,
        silhouette: float | None,
    ) -> None:
        """Agrega una medición comparable al historial interno."""
        self._historial.append(
            {
                "algoritmo": algoritmo,
                "numero_clusters": n_clusters,
                "silhouette": silhouette,
            }
        )

    @abstractmethod
    def ajustar(self, *args, **kwargs):
        """Ajusta el algoritmo implementado por la subclase."""
