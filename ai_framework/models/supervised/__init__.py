"""Supervised learning models: classification and regression."""

from __future__ import annotations

import sys

from . import classification, regression, supervised
from .classification import Classification
from .regression import Regression
from .supervised import Supervised

clasificacion = classification
regresion = regression
from ai_framework.utils.supervised import (
    cross_validation,
    experiments,
    regression_algorithms,
)

algoritmos_regresion = regression_algorithms
experimentos = experiments
experimentos_regresion = experiments
validacion_cruzada = cross_validation

sys.modules[__name__ + ".regression_algorithms"] = regression_algorithms
sys.modules[__name__ + ".experiments"] = experiments
sys.modules[__name__ + ".cross_validation"] = cross_validation
sys.modules[__name__ + ".algoritmos_regresion"] = regression_algorithms
sys.modules[__name__ + ".experimentos"] = experiments
sys.modules[__name__ + ".experimentos_regresion"] = experiments
sys.modules[__name__ + ".validacion_cruzada"] = cross_validation


def __getattr__(name: str):
    if name in ("Clasificacion", "Regresion", "Supervisado"):
        import ai_framework.utils.aliases as aliases
        return getattr(aliases, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Supervised",
    "Classification",
    "Regression",
    "Clasificacion",
    "Regresion",
    "Supervisado",
    "regression_algorithms",
    "experiments",
    "cross_validation",
    "algoritmos_regresion",
    "experimentos",
    "experimentos_regresion",
    "validacion_cruzada",
]
