"""Carga, calidad, partición y preprocesamiento funcional de datos."""

import sys
from .. import datasets

diabetes = datasets.diabetes
concrete = datasets.concrete
potabilidad = datasets.potabilidad

sys.modules.setdefault("framework_ia.datos.diabetes", diabetes)
sys.modules.setdefault("framework_ia.datos.concrete", concrete)
sys.modules.setdefault("framework_ia.datos.potabilidad", potabilidad)

from .dataframe import DataFrame
from .eda import EDA
from .fuentes import CargadorCSV, ConfiguracionCSV
from .particion import (
    ConfiguracionParticion,
    Particionador,
    comprobar_clases,
    dividir_indices,
    huella_datos,
    validar_particion,
)
from .preprocesamiento import (
    ConfiguracionPreprocesamiento,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
)

__all__ = [
    "CargadorCSV",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "DataFrame",
    "EDA",
    "Particionador",
    "PreprocesadorNoSupervisado",
    "PreprocesadorSupervisado",
    "comprobar_clases",
    "dividir_indices",
    "huella_datos",
    "validar_particion",
]
