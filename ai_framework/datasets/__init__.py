"""Dataset preparation, reference, and metadata modules."""

from __future__ import annotations

import sys

from . import concrete, diabetes, potability

potabilidad = potability
sys.modules.setdefault("ai_framework.datasets.potabilidad", potability)

from .concrete import (
    ARCHIVO_CONCRETE,
    ARCHIVO_ORIGINAL_CONCRETE,
    CABECERAS_ORIGINALES_CONCRETE,
    COLUMNAS_CONCRETE,
    FILAS_ORIGINALES_CONCRETE,
    FUENTE_CONCRETE,
    OBJETIVO_CONCRETE,
    PREDICTORAS_CONCRETE,
    UNIDADES_CONCRETE,
    preparar_concrete,
    validar_preparacion_concrete,
)
from .diabetes import (
    ARCHIVO_DIABETES_REGRESION,
    CATEGORICAS_DIABETES,
    COLUMNAS_ORIGINALES,
    FUENTE_DIABETES,
    FUENTE_DIABETES_REGRESION,
    PREDICTORAS_DIABETES,
    cargar_diabetes_regresion,
    preparar_diabetes,
    validar_preparacion_diabetes,
)
from .potability import (
    FUENTE_POTABILIDAD,
    OBJETIVO_POTABILIDAD,
    PREDICTORAS_POTABILIDAD,
    preparar_potabilidad,
    validar_preparacion_potabilidad,
)

potabilidad = sys.modules.get("ai_framework.datasets.potability")

__all__ = [
    "ARCHIVO_CONCRETE",
    "ARCHIVO_DIABETES_REGRESION",
    "ARCHIVO_ORIGINAL_CONCRETE",
    "CABECERAS_ORIGINALES_CONCRETE",
    "CATEGORICAS_DIABETES",
    "COLUMNAS_CONCRETE",
    "COLUMNAS_ORIGINALES",
    "FILAS_ORIGINALES_CONCRETE",
    "FUENTE_CONCRETE",
    "FUENTE_DIABETES",
    "FUENTE_DIABETES_REGRESION",
    "FUENTE_POTABILIDAD",
    "OBJETIVO_CONCRETE",
    "OBJETIVO_POTABILIDAD",
    "PREDICTORAS_CONCRETE",
    "PREDICTORAS_DIABETES",
    "PREDICTORAS_POTABILIDAD",
    "UNIDADES_CONCRETE",
    "cargar_diabetes_regresion",
    "preparar_concrete",
    "preparar_diabetes",
    "preparar_potabilidad",
    "validar_preparacion_concrete",
    "validar_preparacion_diabetes",
    "validar_preparacion_potabilidad",
]
