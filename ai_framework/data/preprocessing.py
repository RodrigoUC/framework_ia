"""Data preprocessing module."""

from framework_ia.datos.preprocesamiento import (
    ConfiguracionPreprocesamiento,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
)

SupervisedPreprocessor = PreprocesadorSupervisado
UnsupervisedPreprocessor = PreprocesadorNoSupervisado
PreprocessingConfig = ConfiguracionPreprocesamiento

__all__ = [
    "SupervisedPreprocessor",
    "UnsupervisedPreprocessor",
    "PreprocessingConfig",
    "PreprocesadorSupervisado",
    "PreprocesadorNoSupervisado",
    "ConfiguracionPreprocesamiento",
]
