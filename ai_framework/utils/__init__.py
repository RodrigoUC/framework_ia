"""Shared functions and configuration objects for AI Framework."""

from __future__ import annotations

import sys

from .cache_numba import configurar_cache_numba
from .colors import color_texto_por_luminancia
from .sklearn_compatibility import crear_codificador_denso, crear_tsne
from .data import (
    ConfiguracionCSV,
    ConfiguracionParticion,
    ConfiguracionPreprocesamiento,
    CSVConfig,
    PartitionConfig,
    PreprocessingConfig,
)
from .unsupervised import (
    ALGORITMOS_CLUSTERING_COMPARACION,
    ModeloKMedoids,
    calcular_centroides,
    comparar_algoritmos_clustering,
    construir_resultado_cluster,
    despachar_algoritmo,
    evaluar_hac,
    evaluar_kmeans,
    validar_numero_clusters,
)
from .supervised import (
    ALGORITMOS_CLASIFICACION,
    ALGORITMOS_LAB2,
    ALGORITMOS_LAB3,
    ALGORITMOS_REGRESION,
    ALPHAS_LAB3,
    ALPHAS_REGRESION,
    CONFIGURACIONES_MODELO,
    CONFIGURACIONES_REGRESION,
    ConfiguracionModelo,
    ConfiguracionRegresion,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
    NOMBRES_REGRESORES,
    ParametroModelo,
    algoritmos_regresion,
    classification as utils_classification,
    comparison as utils_comparison,
    comparar_algoritmos_supervisados,
    configuraciones_experimento,
    configuraciones_experimento_regresion,
    cross_validation as utils_cross_validation,
    despachar_algoritmo_supervisado,
    evidence,
    evidencia,
    experiments as utils_experiments,
    export,
    exportacion,
    regression as utils_regression,
)
from . import data, results, supervised, unsupervised, visualization


def __getattr__(name: str):
    if name in (
        "Clasificacion",
        "Regresion",
        "Supervisado",
        "NoSupervisado",
        "Cluster",
        "DependenciaOpcionalError",
        "aliases",
    ):
        from . import aliases
        return getattr(aliases, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Compatibility aliases
datos = data
no_supervisado = unsupervised
supervisado = supervised
resultados = results
visualizacion = visualization

sys.modules.setdefault("ai_framework.utils.data", data)
sys.modules.setdefault("ai_framework.utils.datos", data)
sys.modules.setdefault("ai_framework.utils.unsupervised", unsupervised)
sys.modules.setdefault("ai_framework.utils.no_supervisado", unsupervised)
sys.modules.setdefault("ai_framework.utils.supervised", supervised)
sys.modules.setdefault("ai_framework.utils.supervisado", supervised)
sys.modules.setdefault("ai_framework.utils.results", results)
sys.modules.setdefault("ai_framework.utils.resultados", results)
sys.modules.setdefault("ai_framework.utils.visualization", visualization)
sys.modules.setdefault("ai_framework.utils.visualizacion", visualization)

from .results import (
    ClassificationExperimentResult,
    ClassificationResult,
    ClusterResult,
    DatosPreparados,
    PartitionResult,
    PCAResult,
    PreparedData,
    ProjectionResult,
    RegressionExperimentResult,
    RegressionResult,
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoExperimentoClasificacion,
    ResultadoExperimentoRegresion,
    ResultadoParticion,
    ResultadoProyeccion,
    ResultadoRegresion,
)
from .visualization import (
    ClassificationVisualizer,
    DataVisualizer,
    RegressionVisualizer,
    SupervisedVisualizer,
    UnsupervisedVisualizer,
    VisualizadorDatos,
    VisualizadorNoSupervisado,
    VisualizadorRegresion,
    VisualizadorSupervisado,
)

__all__ = [
    "configurar_cache_numba",
    "color_texto_por_luminancia",
    "crear_codificador_denso",
    "crear_tsne",
    "CSVConfig",
    "PartitionConfig",
    "PreprocessingConfig",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "ALGORITMOS_CLUSTERING_COMPARACION",
    "ModeloKMedoids",
    "calcular_centroides",
    "comparar_algoritmos_clustering",
    "construir_resultado_cluster",
    "despachar_algoritmo",
    "evaluar_hac",
    "evaluar_kmeans",
    "validar_numero_clusters",
    "ALGORITMOS_CLASIFICACION",
    "ALGORITMOS_LAB2",
    "ALGORITMOS_LAB3",
    "ALGORITMOS_REGRESION",
    "ALPHAS_LAB3",
    "ALPHAS_REGRESION",
    "CONFIGURACIONES_MODELO",
    "CONFIGURACIONES_REGRESION",
    "ConfiguracionModelo",
    "ConfiguracionRegresion",
    "METRICAS_SELECCION",
    "NOMBRES_MODELOS",
    "NOMBRES_REGRESORES",
    "ParametroModelo",
    "algoritmos_regresion",
    "utils_classification",
    "utils_comparison",
    "comparar_algoritmos_supervisados",
    "configuraciones_experimento",
    "configuraciones_experimento_regresion",
    "despachar_algoritmo_supervisado",
    "evidence",
    "export",
    "evidencia",
    "exportacion",
    "utils_experiments",
    "utils_regression",
    "utils_cross_validation",
    "data",
    "unsupervised",
    "supervised",
    "results",
    "visualization",
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
]
