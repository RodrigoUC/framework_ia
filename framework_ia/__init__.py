"""Framework modular de análisis exploratorio y aprendizaje automático (Nexo IA)."""

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
from . import datasets
from .modelos import (
    Clasificacion,
    Cluster,
    NoSupervisado,
    Progresion,
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
    "Progresion",
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
