"""Objetos de resultado compartidos por los análisis no supervisados.

Las clases de este módulo transportan datos entre la lógica de modelos y la
capa de visualización. No conocen Streamlit ni generan efectos secundarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatosPreparados:
    """Matriz numérica lista para modelos y metadatos del preprocesamiento."""

    matriz: pd.DataFrame
    columnas_origen: tuple[str, ...]
    columnas_descartadas: tuple[str, ...]
    transformador: Any = field(repr=False)


@dataclass(frozen=True)
class ResultadoACP:
    """Resultado completo de un análisis de componentes principales."""

    coordenadas: pd.DataFrame
    cargas: pd.DataFrame
    varianza_explicada: pd.Series
    varianza_acumulada: pd.Series
    datos_preparados: DatosPreparados
    modelo: Any = field(repr=False)


@dataclass(frozen=True)
class ResultadoProyeccion:
    """Coordenadas bidimensionales producidas por t-SNE o UMAP."""

    algoritmo: str
    coordenadas: pd.DataFrame
    datos_preparados: DatosPreparados
    modelo: Any = field(repr=False)


@dataclass(frozen=True)
class ResultadoCluster:
    """Resultado uniforme para algoritmos de agrupamiento."""

    algoritmo: str
    etiquetas: pd.Series
    centroides: pd.DataFrame
    proyeccion_2d: pd.DataFrame
    silhouette: float | None
    datos_preparados: DatosPreparados
    modelo: Any = field(repr=False)
    inercia: float | None = None
    matriz_vinculacion: Any | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ResultadoParticion:
    """Partición reproducible de un dataset en train/test y validación opcional.

    Se calcula una sola vez sobre el dataset activo (vista Datos) y puede
    reutilizarse en Clasificación o Regresión en lugar de generar una
    partición aleatoria distinta en cada vista.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    validacion: pd.DataFrame | None
    porcentaje_train: float
    porcentaje_test: float
    porcentaje_validacion: float
    columna_estratificacion: str | None
    distribucion_train: dict[Any, int] | None
    distribucion_test: dict[Any, int] | None
    distribucion_validacion: dict[Any, int] | None
    semilla: int
    advertencia: str | None = None

    @property
    def huella(self) -> tuple:
        """Identifica la partición para invalidar resultados dependientes."""
        return (
            tuple(self.train.index),
            tuple(self.test.index),
            tuple(self.validacion.index) if self.validacion is not None else (),
            self.columna_estratificacion,
            self.semilla,
        )


@dataclass(frozen=True)
class ResultadoClasificacion:
    """Salida de entrenamiento y evaluación supervisada para clasificación."""

    algoritmo: str
    target: str
    features: tuple[str, ...]
    labels: tuple[Any, ...]
    y_true: pd.Series
    y_pred: pd.Series
    matriz_confusion: pd.DataFrame
    metricas: dict[str, Any]
    muestra_train: int
    muestra_test: int
    modelo: Any = field(repr=False)
    parametros: dict[str, Any] = field(default_factory=dict)
    metadatos: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultadoExperimentoClasificacion:
    """Selección por validación y evaluación final de los ganadores en prueba.

    ``tabla`` nunca contiene métricas de prueba de candidatos descartados.
    ``mejor_algoritmo`` se decide exclusivamente por la métrica de validación.
    """

    tabla: pd.DataFrame
    mejores: dict[str, ResultadoClasificacion]
    mejor_algoritmo: str | None
    metadatos: dict[str, Any]
