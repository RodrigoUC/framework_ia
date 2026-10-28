"""Framework modular de análisis exploratorio y aprendizaje automático (Nexo IA)."""

import sys

from . import datasets
from .datos import (
    CargadorCSV,
    ConfiguracionCSV,
    ConfiguracionParticion,
    ConfiguracionPreprocesamiento,
    DataFrame,
    EDA,
    Particionador,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
)
from .modelos import (
    Clasificacion,
    Cluster,
    NoSupervisado,
    Regresion,
    Supervisado,
)
from .utils import (
    evidencia,
    exportacion,
    resultados,
    visualizacion,
)
from .utils.resultados import (
    DatosPreparados,
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoExperimentoClasificacion,
    ResultadoExperimentoRegresion,
    ResultadoParticion,
    ResultadoProyeccion,
    ResultadoRegresion,
)

# Alias en sys.modules para mantener compatibilidad histórica con framework_ia.<modulo>
sys.modules.setdefault("framework_ia.resultados", resultados)
sys.modules.setdefault("framework_ia.visualizacion", visualizacion)
sys.modules.setdefault("framework_ia.exportacion", exportacion)
sys.modules.setdefault("framework_ia.evidencia", evidencia)

__all__ = [
    "CargadorCSV",
    "Clasificacion",
    "Cluster",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "DataFrame",
    "DatosPreparados",
    "EDA",
    "NoSupervisado",
    "Particionador",
    "PreprocesadorNoSupervisado",
    "PreprocesadorSupervisado",
    "Regresion",
    "ResultadoACP",
    "ResultadoClasificacion",
    "ResultadoCluster",
    "ResultadoExperimentoClasificacion",
    "ResultadoExperimentoRegresion",
    "ResultadoParticion",
    "ResultadoProyeccion",
    "ResultadoRegresion",
    "Supervisado",
    "datasets",
    "evidencia",
    "exportacion",
    "resultados",
    "visualizacion",
]
