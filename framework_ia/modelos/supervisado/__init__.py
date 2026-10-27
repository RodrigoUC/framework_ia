"""Modelos supervisados: clasificación y regresión."""

from .base import Supervisado
from .clasificacion import Clasificacion
from .regresion import Regresion

__all__ = ["Supervisado", "Clasificacion", "Regresion"]
