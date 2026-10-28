"""Supervised learning models."""

from .classification import Clasificacion, Classification
from .regression import Regresion, Regression
from .supervised import Supervisado, Supervised

__all__ = [
    "Classification",
    "Regression",
    "Supervised",
    "Clasificacion",
    "Regresion",
    "Supervisado",
]
