"""Framework modular de análisis exploratorio y aprendizaje automático (Nexo IA)."""

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
from .resultados import (
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
]
