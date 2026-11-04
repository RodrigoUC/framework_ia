"""Data loading, quality, partitioning, and preprocessing module."""

from __future__ import annotations

import sys
from .. import datasets

diabetes = datasets.diabetes
concrete = datasets.concrete
potability = datasets.potability
potabilidad = potability

sys.modules.setdefault("ai_framework.data.diabetes", diabetes)
sys.modules.setdefault("ai_framework.data.concrete", concrete)
sys.modules.setdefault("ai_framework.data.potability", potability)
sys.modules.setdefault("ai_framework.data.potabilidad", potabilidad)

from .dataframe import DataFrame
from .eda import EDA
from .sources import CargadorCSV, ConfiguracionCSV, CSVConfig, CSVLoader
from .partition import (
    ConfiguracionParticion,
    DataSplitter,
    Particionador,
    PartitionConfig,
    comprobar_clases,
    dividir_indices,
    huella_datos,
    validar_particion,
)
from .preprocessing import (
    ConfiguracionPreprocesamiento,
    PreprocessingConfig,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
    SupervisedPreprocessor,
    UnsupervisedPreprocessor,
)

__all__ = [
    # English names
    "DataFrame",
    "EDA",
    "CSVLoader",
    "CSVConfig",
    "DataSplitter",
    "PartitionConfig",
    "SupervisedPreprocessor",
    "UnsupervisedPreprocessor",
    "PreprocessingConfig",
    # Bilingual aliases
    "CargadorCSV",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "Particionador",
    "PreprocesadorNoSupervisado",
    "PreprocesadorSupervisado",
    "ConfiguracionPreprocesamiento",
    "comprobar_clases",
    "dividir_indices",
    "huella_datos",
    "validar_particion",
]
