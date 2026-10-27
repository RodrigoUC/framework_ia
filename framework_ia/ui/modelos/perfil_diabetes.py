"""Modelo de validación de procedencia para el CSV preparado de Diabetes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from framework_ia.datos.diabetes import PREDICTORAS_DIABETES
from framework_ia.ui.modelos.catalogo_vistas import BASE_DIR

RUTA_PROCEDENCIA = BASE_DIR / "data" / "lab02" / "procedencia.json"
_CATEGORICAS = ("location", "gender", "frame")


def estado_fuente_diabetes(datos: pd.DataFrame, estado) -> str | None:
    """Reconoce los bytes registrados de la fuente sin inferir perfiles por nombre."""
    fuente = estado.get("fuente_aplicada")
    metadata = estado.get("fuente_metadata", {})
    if fuente is None or metadata.get("identidad") != fuente[2]:
        return None
    digest = metadata.get("sha256")
    if not digest:
        return None
    try:
        perfil = json.loads(RUTA_PROCEDENCIA.read_text(encoding="utf-8"))["diabetes"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(perfil, dict):
        return None
    original = perfil.get("original", {})
    derivacion = perfil.get("derivacion", {})
    if not isinstance(original, dict) or not isinstance(derivacion, dict):
        return None
    regla = derivacion.get("regla", {})
    if not isinstance(regla, dict):
        return None
    original = original.get("sha256")
    if original and digest == original:
        return "original"
    if digest != perfil.get("sha256"):
        return None
    preparacion = estado.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        return None
    requeridas = (*PREDICTORAS_DIABETES, "diabetes")
    if (
        datos.empty
        or not datos.index.is_unique
        or not datos.columns.is_unique
        or not set(requeridas).issubset(datos.columns)
        or not datos.equals(fuente[0])
    ):
        return None
    objetivo = datos["diabetes"]
    if objetivo.isna().any() or objetivo.nunique() != 2:
        return None
    etiquetas = (regla.get("positiva"), regla.get("negativa"))
    if (
        all(etiqueta is not None for etiqueta in etiquetas)
        and not objetivo.isin(etiquetas).all()
    ):
        return None
    numericas = [
        columna for columna in PREDICTORAS_DIABETES if columna not in _CATEGORICAS
    ]
    if any(not pd.api.types.is_numeric_dtype(datos[columna]) for columna in numericas):
        return None
    if any(pd.api.types.is_numeric_dtype(datos[columna]) for columna in _CATEGORICAS):
        return None
    return "preparado"
