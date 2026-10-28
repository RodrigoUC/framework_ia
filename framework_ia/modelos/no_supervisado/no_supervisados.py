"""Abstracciones base y métodos comunes compartidos por análisis no supervisados."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ...datos.eda import EDA
from ...datos.preprocesamiento import PreprocesadorNoSupervisado
from ...resultados import DatosPreparados
from ...utils import ConfiguracionPreprocesamiento


class DependenciaOpcionalError(ImportError):
    """Indica que una capacidad opcional requiere instalar una dependencia."""


class NoSupervisado(EDA, ABC):
    """Clase base para análisis no supervisados: preprocesamiento y benchmark."""

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
