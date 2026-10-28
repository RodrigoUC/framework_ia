"""Modelos supervisados: clasificación y regresión."""

import sys

from .clasificacion import Clasificacion
from .regresion import Regresion
from .supervisado import Supervisado
from ...utils.supervisado import (
    algoritmos_regresion,
    experimentos,
    validacion_cruzada,
)
experimentos_regresion = experimentos

sys.modules[__name__ + ".algoritmos_regresion"] = algoritmos_regresion
sys.modules[__name__ + ".experimentos"] = experimentos
sys.modules[__name__ + ".experimentos_regresion"] = experimentos
sys.modules[__name__ + ".validacion_cruzada"] = validacion_cruzada

__all__ = [
    "Supervisado",
    "Clasificacion",
    "Regresion",
    "algoritmos_regresion",
    "experimentos",
    "experimentos_regresion",
    "validacion_cruzada",
]
