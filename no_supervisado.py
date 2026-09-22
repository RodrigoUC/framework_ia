"""Abstracciones compartidas por los análisis no supervisados."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from eda import EDA
from preprocesamiento import (
    ConfiguracionPreprocesamiento,
    PreprocesadorNoSupervisado,
)
from resultados import DatosPreparados


class NoSupervisado(EDA, ABC):
    """Base común que delega el preprocesamiento y registra métricas."""

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
        """Obtiene una matriz numérica reutilizable por el algoritmo concreto."""
        if self._datos_preparados is None or refrescar:
            preprocesador = PreprocesadorNoSupervisado(self.configuracion)
            self._datos_preparados = preprocesador.ajustar_transformar(self.datos)
        return self._datos_preparados

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
