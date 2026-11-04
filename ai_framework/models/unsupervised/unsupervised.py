"""Base abstractions and common methods shared by unsupervised analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ai_framework.data.eda import EDA
from ai_framework.data.preprocessing import PreprocesadorNoSupervisado
from ai_framework.utils.results import DatosPreparados
from ai_framework.utils import ConfiguracionPreprocesamiento


class OptionalDependencyError(ImportError):
    """Indicates that an optional capability requires installing a dependency."""


class Unsupervised(EDA):
    """Base class for unsupervised analyses: preprocessing and benchmark."""

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
        """Obtain a reusable numerical matrix for unsupervised analyses."""
        if self._datos_preparados is None or refrescar:
            preprocesador = PreprocesadorNoSupervisado(self.configuracion)
            self._datos_preparados = preprocesador.ajustar_transformar(self.datos)
        return self._datos_preparados

    def proyectar_2d(self, matriz: pd.DataFrame | None = None) -> pd.DataFrame:
        """Project observations into 2 dimensions for visual analysis."""
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
        """Return metrics recorded during the current instance."""
        return pd.DataFrame(self._historial)

    prepare_matrix = preparar_matriz
    project_2d = proyectar_2d

    def _registrar_resultado(
        self,
        algoritmo: str,
        n_clusters: int,
        silhouette: float | None,
    ) -> None:
        """Add a comparable metric to the internal history."""
        self._historial.append(
            {
                "algoritmo": algoritmo,
                "numero_clusters": n_clusters,
                "silhouette": silhouette,
            }
        )


def __getattr__(name: str):
    if name in ("NoSupervisado", "DependenciaOpcionalError"):
        import ai_framework.utils.aliases as aliases
        return getattr(aliases, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
