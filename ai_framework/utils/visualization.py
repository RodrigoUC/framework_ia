"""Visualization utilities."""

from framework_ia.utils.visualizacion import (
    VisualizadorDatos,
    VisualizadorNoSupervisado,
    VisualizadorRegresion,
    VisualizadorSupervisado,
)

SupervisedVisualizer = VisualizadorSupervisado
ClassificationVisualizer = VisualizadorSupervisado
DataVisualizer = VisualizadorDatos
UnsupervisedVisualizer = VisualizadorNoSupervisado
RegressionVisualizer = VisualizadorRegresion

__all__ = [
    "SupervisedVisualizer",
    "ClassificationVisualizer",
    "DataVisualizer",
    "UnsupervisedVisualizer",
    "RegressionVisualizer",
    "VisualizadorSupervisado",
    "VisualizadorDatos",
    "VisualizadorNoSupervisado",
    "VisualizadorRegresion",
]
