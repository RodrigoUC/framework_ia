"""Controller for instantiation, execution context, and classification metrics."""

from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import streamlit as st

from ai_framework.models.supervised import Clasificacion
from ai_framework.ui.models.view_catalog import FOLDS_LAB2
from ai_framework.utils import ALGORITMOS_LAB2


def _common_arguments(configuracion: dict) -> dict:
    """Build common partition and preprocessing arguments for classification."""
    return {
        "test_size": configuracion["test_size"],
        "random_state": configuracion["random_state"],
        "stratify": configuracion["estratificar"],
        "incluir_categoricas": configuracion["incluir_categoricas"],
        "imputar": configuracion["imputar"],
        "estandarizar": configuracion["estandarizar"],
        "particion": configuracion.get("particion_global"),
    }


def _create_model(datos: pd.DataFrame, configuracion: dict) -> Clasificacion:
    """Instantiate classification domain model for the active dataset."""
    return Clasificacion(
        dataframe=datos,
        target=configuracion["target"],
        features=configuracion["features"],
    )


def _execution_context(configuracion: dict) -> dict:
    """Freeze classification execution context for traceability."""
    return {
        "dataset": st.session_state.get("fuente_aplicada", (None, "", ""))[1],
        **{
            clave: valor
            for clave, valor in configuracion.items()
            if clave != "particion_global"
        },
    }


def _json(data) -> str:
    """Serialize parameter or result objects into readable JSON format."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def true_rate_name(label, labels) -> str:
    """Return descriptive name of true rate for a class."""
    labels_list = list(labels)
    if len(labels_list) == 2:
        str_set = {str(e).strip().casefold() for e in labels_list}
        str_label = str(label).strip().casefold()
        if str_set in ({"0", "1"}, {"n", "y"}, {"false", "true"}, {"no", "si"}, {"no", "sí"}):
            is_positive = str_label in {"1", "y", "true", "si", "sí"}
            kind = "Positivos" if is_positive else "Negativos"
            label_fmt = str(label).upper() if str_set == {"n", "y"} else str(label)
            return f"Tasa de Verdaderos {kind} ({label_fmt}) (%)"
    return f"Tasa de Verdaderos ({label}) (%)"


def _metrics_by_class(result) -> dict[str, float | str]:
    """Calculate percentage true rate by actual class on test predictions."""
    y_true = result.y_true
    y_pred = result.y_pred
    labels = list(result.labels)
    recalls = {}
    for label in labels:
        name = true_rate_name(label, labels)
        actual = y_true == label
        support = int(actual.sum())
        recalls[name] = (
            100.0 * int((y_pred[actual] == label).sum()) / support
            if support
            else "Sin casos evaluados"
        )
    return recalls


def _percentage_metrics(result) -> dict[str, float | str]:
    """Combine overall percentage accuracy with true rate by actual class."""
    return {
        "Accuracy general (%)": 100.0 * float(result.metricas["accuracy"]),
        **_metrics_by_class(result),
    }


def _executed_source(data: pd.DataFrame) -> dict:
    """Reconstruct metadata of data source used in the experiment."""
    source = st.session_state.get("fuente_aplicada", (None, "", ""))
    metadata = st.session_state.get("fuente_metadata", {})
    if metadata.get("identidad") != source[2]:
        metadata = {}
    return {
        "archivo": source[1],
        "sha256": metadata.get("sha256"),
        "fuente": metadata.get("fuente"),
        "filas": len(data),
        "columnas": list(data.columns),
        "filas_archivo": metadata.get("filas"),
        "columnas_archivo": metadata.get("columnas"),
        "lectura": deepcopy(metadata.get("lectura", {})),
        "preparacion_previa": deepcopy(
            st.session_state.get("dataset_preparacion", {})
        ),
    }


def _export_arguments(
    config: dict,
    options: dict,
    source: dict,
) -> dict:
    """Build reproducibility arguments dictionary for export."""
    return {
        "csv": source["archivo"],
        "target": config["target"],
        "features": list(config["features"]),
        "semilla": config["random_state"],
        "test_size": config["test_size"],
        "validation_size": options["validation_size"],
        "estrategia": options.get("estrategia", "holdout"),
        "folds": FOLDS_LAB2 if options.get("estrategia") == "cv" else None,
        "metrica": options["metrica"],
        "algoritmos": list(ALGORITMOS_LAB2),
        "incluir_categoricas": config["incluir_categoricas"],
        "sin_imputar": not config["imputar"],
        "sin_escalar": not config["estandarizar"],
        "estratificar": config["estratificar"],
        "particion_externa": config.get("particion_global") is not None,
        **source["lectura"],
        "fuente": source["fuente"],
    }

nombre_tasa_verdaderos = true_rate_name

# Backward compatibility aliases
_argumentos_comunes = _common_arguments
_modelo = _create_model
_contexto_ejecucion = _execution_context
_metricas_por_clase = _metrics_by_class
_metricas_porcentuales = _percentage_metrics
_fuente_ejecutada = _executed_source
_argumentos_exportacion = _export_arguments
