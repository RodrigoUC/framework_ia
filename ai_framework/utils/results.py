"""Objetos de resultado compartidos por los análisis no supervisados y supervisados.

Las clases de este módulo transportan datos entre la lógica de modelos y la
capa de visualización. No conocen Streamlit ni generan efectos secundarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    """Numerical matrix ready for models and preprocessing metadata."""

    matriz: pd.DataFrame
    columnas_origen: tuple[str, ...]
    columnas_descartadas: tuple[str, ...]
    transformador: Any = field(repr=False)

    @property
    def matrix(self) -> pd.DataFrame:
        return self.matriz

    @property
    def source_columns(self) -> tuple[str, ...]:
        return self.columnas_origen

    @property
    def discarded_columns(self) -> tuple[str, ...]:
        return self.columnas_descartadas

    @property
    def transformer(self) -> Any:
        return self.transformador


DatosPreparados = PreparedData


@dataclass(frozen=True)
class PCAResult:
    """Full outcome of a principal components analysis."""

    coordenadas: pd.DataFrame
    cargas: pd.DataFrame
    varianza_explicada: pd.Series
    varianza_acumulada: pd.Series
    datos_preparados: DatosPreparados
    modelo: Any = field(repr=False)

    @property
    def coordinates(self) -> pd.DataFrame:
        return self.coordenadas

    @property
    def loadings(self) -> pd.DataFrame:
        return self.cargas

    @property
    def explained_variance(self) -> pd.Series:
        return self.varianza_explicada

    @property
    def cumulative_variance(self) -> pd.Series:
        return self.varianza_acumulada

    @property
    def prepared_data(self) -> DatosPreparados:
        return self.datos_preparados

    @property
    def model(self) -> Any:
        return self.modelo


ResultadoACP = PCAResult


@dataclass(frozen=True)
class ProjectionResult:
    """Coordenadas bidimensionales producidas por t-SNE o UMAP."""

    algoritmo: str
    coordenadas: pd.DataFrame
    datos_preparados: DatosPreparados
    modelo: Any = field(repr=False)

    @property
    def algorithm(self) -> str:
        return self.algoritmo

    @property
    def coordinates(self) -> pd.DataFrame:
        return self.coordenadas

    @property
    def prepared_data(self) -> DatosPreparados:
        return self.datos_preparados

    @property
    def model(self) -> Any:
        return self.modelo


ResultadoProyeccion = ProjectionResult


@dataclass(frozen=True)
class ClusterResult:
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

    @property
    def algorithm(self) -> str:
        return self.algoritmo

    @property
    def labels(self) -> pd.Series:
        return self.etiquetas

    @property
    def centroids(self) -> pd.DataFrame:
        return self.centroides

    @property
    def projection_2d(self) -> pd.DataFrame:
        return self.proyeccion_2d

    @property
    def prepared_data(self) -> DatosPreparados:
        return self.datos_preparados

    @property
    def model(self) -> Any:
        return self.modelo

    @property
    def inertia(self) -> float | None:
        return self.inercia

    @property
    def linkage_matrix(self) -> Any | None:
        return self.matriz_vinculacion


ResultadoCluster = ClusterResult


@dataclass(frozen=True)
class PartitionResult:
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
    def validation(self) -> pd.DataFrame | None:
        return self.validacion

    @property
    def train_percentage(self) -> float:
        return self.porcentaje_train

    @property
    def test_percentage(self) -> float:
        return self.porcentaje_test

    @property
    def validation_percentage(self) -> float:
        return self.porcentaje_validacion

    @property
    def stratify_column(self) -> str | None:
        return self.columna_estratificacion

    @property
    def seed(self) -> int:
        return self.semilla

    @property
    def warning(self) -> str | None:
        return self.advertencia

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


ResultadoParticion = PartitionResult


@dataclass(frozen=True)
class ClassificationResult:
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

    @property
    def algorithm(self) -> str:
        return self.algoritmo

    @property
    def confusion_matrix(self) -> pd.DataFrame:
        return self.matriz_confusion

    @property
    def metrics(self) -> dict[str, Any]:
        return self.metricas

    @property
    def parameters(self) -> dict[str, Any]:
        return self.parametros

    @property
    def metadata(self) -> dict[str, Any]:
        return self.metadatos

    @property
    def model(self) -> Any:
        return self.modelo


ResultadoClasificacion = ClassificationResult


@dataclass(frozen=True)
class ClassificationExperimentResult:
    """Selección por validación y evaluación final de los ganadores en prueba.

    ``tabla`` nunca contiene métricas de prueba de candidatos descartados.
    ``mejor_algoritmo`` se decide exclusivamente por la métrica de validación.
    """

    tabla: pd.DataFrame
    mejores: dict[str, ClassificationResult]
    mejor_algoritmo: str | None
    metadatos: dict[str, Any]

    @property
    def table(self) -> pd.DataFrame:
        return self.tabla

    @property
    def best_by_family(self) -> dict[str, ClassificationResult]:
        return self.mejores

    @property
    def best_algorithm(self) -> str | None:
        return self.mejor_algoritmo

    @property
    def metadata(self) -> dict[str, Any]:
        return self.metadatos


ResultadoExperimentoClasificacion = ClassificationExperimentResult


@dataclass(frozen=True)
class RegressionResult:
    """Evaluación numérica y artefactos ajustados sólo con entrenamiento.

    ``y_true``, ``y_pred``, ``residuos`` y ``metricas`` corresponden al conjunto
    de evaluación indicado por ``metadatos``; los campos train son separados.
    El contenedor es frozen, como los demás resultados; pandas/modelos internos
    siguen siendo objetos mutables y deben tratarse como snapshots de lectura.
    """

    algoritmo: str
    target: str
    features: tuple[str, ...]
    y_true: pd.Series
    y_pred: pd.Series
    y_train: pd.Series
    y_pred_train: pd.Series
    residuos: pd.Series
    residuos_train: pd.Series
    metricas: dict[str, float]
    metricas_train: dict[str, float]
    muestra_train: int
    muestra_test: int
    modelo: Any = field(repr=False)
    datos_preparados: DatosPreparados = field(repr=False)
    parametros: dict[str, Any] = field(default_factory=dict)
    metadatos: dict[str, Any] = field(default_factory=dict)

    @property
    def algorithm(self) -> str:
        return self.algoritmo

    @property
    def residuals(self) -> pd.Series:
        return self.residuos

    @property
    def residuals_train(self) -> pd.Series:
        return self.residuos_train

    @property
    def metrics(self) -> dict[str, float]:
        return self.metricas

    @property
    def metrics_train(self) -> dict[str, float]:
        return self.metricas_train

    @property
    def model(self) -> Any:
        return self.modelo

    @property
    def prepared_data(self) -> DatosPreparados:
        return self.datos_preparados

    @property
    def parameters(self) -> dict[str, Any]:
        return self.parametros

    @property
    def metadata(self) -> dict[str, Any]:
        return self.metadatos


ResultadoRegresion = RegressionResult


@dataclass(frozen=True)
class RegressionExperimentResult:
    """Comparación por validación; sólo los ganadores tienen evaluación en prueba."""

    tabla: pd.DataFrame
    mejores: dict[str, RegressionResult]
    mejor_algoritmo: str | None
    metadatos: dict[str, Any]

    @property
    def table(self) -> pd.DataFrame:
        return self.tabla

    @property
    def best_by_family(self) -> dict[str, RegressionResult]:
        return self.mejores

    @property
    def best_algorithm(self) -> str | None:
        return self.mejor_algoritmo

    @property
    def metadata(self) -> dict[str, Any]:
        return self.metadatos


ResultadoExperimentoRegresion = RegressionExperimentResult
