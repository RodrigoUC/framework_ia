"""Results dataclasses and exports."""

from framework_ia.utils.resultados import (
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

PreparedData = DatosPreparados
PCAResult = ResultadoACP
ClassificationResult = ResultadoClasificacion
ClusterResult = ResultadoCluster
ClassificationExperimentResult = ResultadoExperimentoClasificacion
RegressionExperimentResult = ResultadoExperimentoRegresion
PartitionResult = ResultadoParticion
ProjectionResult = ResultadoProyeccion
RegressionResult = ResultadoRegresion

__all__ = [
    "PreparedData",
    "PCAResult",
    "ClassificationResult",
    "ClusterResult",
    "ClassificationExperimentResult",
    "RegressionExperimentResult",
    "PartitionResult",
    "ProjectionResult",
    "RegressionResult",
    "DatosPreparados",
    "ResultadoACP",
    "ResultadoClasificacion",
    "ResultadoCluster",
    "ResultadoExperimentoClasificacion",
    "ResultadoExperimentoRegresion",
    "ResultadoParticion",
    "ResultadoProyeccion",
    "ResultadoRegresion",
]
