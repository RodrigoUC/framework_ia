"""Modular framework for exploratory data analysis and machine learning (Nexo AI / AI Framework)."""

from __future__ import annotations

import sys

from . import data, datasets, models, ui, utils
from .data import (
    CargadorCSV,
    ConfiguracionCSV,
    ConfiguracionParticion,
    ConfiguracionPreprocesamiento,
    CSVConfig,
    CSVLoader,
    DataFrame,
    DataSplitter,
    EDA,
    Particionador,
    PartitionConfig,
    PreprocessingConfig,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
    SupervisedPreprocessor,
    UnsupervisedPreprocessor,
)
from .models import (
    Clasificacion,
    Classification,
    Cluster,
    Clustering,
    NoSupervisado,
    Regresion,
    Regression,
    Supervisado,
    Supervised,
    Unsupervised,
)
from .utils import (
    ClassificationExperimentResult,
    ClassificationResult,
    ClassificationVisualizer,
    ClusterResult,
    DataVisualizer,
    DatosPreparados,
    PartitionResult,
    PCAResult,
    PreparedData,
    ProjectionResult,
    RegressionExperimentResult,
    RegressionResult,
    RegressionVisualizer,
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoExperimentoClasificacion,
    ResultadoExperimentoRegresion,
    ResultadoParticion,
    ResultadoProyeccion,
    ResultadoRegresion,
    SupervisedVisualizer,
    UnsupervisedVisualizer,
    VisualizadorDatos,
    VisualizadorNoSupervisado,
    VisualizadorRegresion,
    VisualizadorSupervisado,
    evidence,
    evidencia,
    export,
    exportacion,
    resultados,
    results,
    visualizacion,
    visualization,
)

# Compatibility aliases
sys.modules.setdefault("ai_framework.results", results)
sys.modules.setdefault("ai_framework.resultados", results)
sys.modules.setdefault("ai_framework.visualization", visualization)
sys.modules.setdefault("ai_framework.visualizacion", visualization)
sys.modules.setdefault("ai_framework.export", export)
sys.modules.setdefault("ai_framework.exportacion", export)
sys.modules.setdefault("ai_framework.evidence", evidence)
sys.modules.setdefault("ai_framework.evidencia", evidence)

# Also expose ai_framework.datos and ai_framework.modelos aliases if imported as submodules
sys.modules.setdefault("ai_framework.datos", data)
sys.modules.setdefault("ai_framework.modelos", models)

__all__ = [
    # English Primary API
    "Classification",
    "Regression",
    "Clustering",
    "Supervised",
    "Unsupervised",
    "DataFrame",
    "EDA",
    "CSVLoader",
    "CSVConfig",
    "DataSplitter",
    "PartitionConfig",
    "SupervisedPreprocessor",
    "UnsupervisedPreprocessor",
    "PreprocessingConfig",
    "PreparedData",
    "PCAResult",
    "ClassificationResult",
    "ClusterResult",
    "ClassificationExperimentResult",
    "RegressionExperimentResult",
    "PartitionResult",
    "ProjectionResult",
    "RegressionResult",
    "SupervisedVisualizer",
    "ClassificationVisualizer",
    "DataVisualizer",
    "UnsupervisedVisualizer",
    "RegressionVisualizer",
    # Subpackages
    "data",
    "models",
    "datasets",
    "ui",
    "utils",
    # Bilingual Aliases
    "Clasificacion",
    "Regresion",
    "Cluster",
    "Supervisado",
    "NoSupervisado",
    "CargadorCSV",
    "ConfiguracionCSV",
    "Particionador",
    "ConfiguracionParticion",
    "PreprocesadorSupervisado",
    "PreprocesadorNoSupervisado",
    "ConfiguracionPreprocesamiento",
    "DatosPreparados",
    "ResultadoACP",
    "ResultadoClasificacion",
    "ResultadoCluster",
    "ResultadoExperimentoClasificacion",
    "ResultadoExperimentoRegresion",
    "ResultadoParticion",
    "ResultadoProyeccion",
    "ResultadoRegresion",
    "VisualizadorSupervisado",
    "VisualizadorDatos",
    "VisualizadorNoSupervisado",
    "VisualizadorRegresion",
    "resultados",
    "results",
    "visualizacion",
    "visualization",
    "exportacion",
    "export",
    "evidencia",
    "evidence",
]
