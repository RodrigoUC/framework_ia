"""Provenance validation model for the prepared Diabetes CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_framework.datasets.diabetes import PREDICTORAS_DIABETES
from ai_framework.ui.models.view_catalog import BASE_DIR

RUTA_PROCEDENCIA = BASE_DIR / "data" / "lab02" / "procedencia.json"
_CATEGORICAS = ("location", "gender", "frame")


def diabetes_source_state(data: pd.DataFrame, state) -> str | None:
    """Recognize recorded source bytes without inferring profiles by filename."""
    source = state.get("fuente_aplicada")
    metadata = state.get("fuente_metadata", {})
    if source is None or metadata.get("identidad") != source[2]:
        return None
    digest = metadata.get("sha256")
    if not digest:
        return None
    try:
        profile = json.loads(RUTA_PROCEDENCIA.read_text(encoding="utf-8"))["diabetes"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(profile, dict):
        return None
    original = profile.get("original", {})
    derivation = profile.get("derivacion", {})
    if not isinstance(original, dict) or not isinstance(derivation, dict):
        return None
    rule = derivation.get("regla", {})
    if not isinstance(rule, dict):
        return None
    original = original.get("sha256")
    if original and digest == original:
        return "original"
    if digest != profile.get("sha256"):
        return None
    preparation = state.get("dataset_preparacion", {})
    if (
        preparation.get("imputar")
        or preparation.get("escalado", "Ninguno") != "Ninguno"
    ):
        return None
    required = (*PREDICTORAS_DIABETES, "diabetes")
    if (
        data.empty
        or not data.index.is_unique
        or not data.columns.is_unique
        or not set(required).issubset(data.columns)
        or not data.equals(source[0])
    ):
        return None
    target = data["diabetes"]
    if target.isna().any() or target.nunique() != 2:
        return None
    labels = (rule.get("positiva"), rule.get("negativa"))
    if (
        all(label is not None for label in labels)
        and not target.isin(labels).all()
    ):
        return None
    numeric = [
        col for col in PREDICTORAS_DIABETES if col not in _CATEGORICAS
    ]
    if any(not pd.api.types.is_numeric_dtype(data[col]) for col in numeric):
        return None
    if any(pd.api.types.is_numeric_dtype(data[col]) for col in _CATEGORICAS):
        return None
    return "preparado"


# Backward compatibility aliases
estado_fuente_diabetes = diabetes_source_state
