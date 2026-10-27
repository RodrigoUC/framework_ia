"""Controlador de instanciación, contexto de ejecución y métricas de clasificación."""

from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import streamlit as st

from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.ui.modelos.catalogo_vistas import FOLDS_LAB2
from framework_ia.utils import ALGORITMOS_LAB2


def _argumentos_comunes(configuracion: dict) -> dict:
    """Construye los argumentos comunes de partición y preprocesamiento para clasificación."""
    return {
        "test_size": configuracion["test_size"],
        "random_state": configuracion["random_state"],
        "stratify": configuracion["estratificar"],
        "incluir_categoricas": configuracion["incluir_categoricas"],
        "imputar": configuracion["imputar"],
        "estandarizar": configuracion["estandarizar"],
        "particion": configuracion.get("particion_global"),
    }


def _modelo(datos: pd.DataFrame, configuracion: dict) -> Clasificacion:
    """Instancia el modelo de dominio de clasificación para el dataset activo."""
    return Clasificacion(
        dataframe=datos,
        target=configuracion["target"],
        features=configuracion["features"],
    )


def _contexto_ejecucion(configuracion: dict) -> dict:
    """Congela el contexto de ejecución de clasificación para trazabilidad."""
    return {
        "dataset": st.session_state.get("fuente_aplicada", (None, "", ""))[1],
        **{
            clave: valor
            for clave, valor in configuracion.items()
            if clave != "particion_global"
        },
    }


def _json(datos) -> str:
    """Serializa parámetros u objetos de resultados en formato JSON legible."""
    return json.dumps(datos, ensure_ascii=False, indent=2, default=str)


def _metricas_por_clase(resultado) -> dict[str, float | str]:
    """Calcula el recall porcentual por clase real sobre las predicciones de prueba."""
    y_true = resultado.y_true
    y_pred = resultado.y_pred
    etiquetas = list(resultado.labels)
    binarias_yn = (
        len(etiquetas) == 2
        and {str(etiqueta).casefold() for etiqueta in etiquetas} == {"y", "n"}
        and all(isinstance(etiqueta, str) for etiqueta in etiquetas)
    )
    recalls = {}
    for etiqueta in etiquetas:
        nombre = (
            f"Recall {str(etiqueta).upper()} (%)"
            if binarias_yn
            else f"Recall actual {etiqueta!s} (%)"
        )
        reales = y_true == etiqueta
        soporte = int(reales.sum())
        recalls[nombre] = (
            100.0 * int((y_pred[reales] == etiqueta).sum()) / soporte
            if soporte
            else "Sin casos evaluados"
        )
    return recalls


def _metricas_porcentuales(resultado) -> dict[str, float | str]:
    """Combina el accuracy porcentual general con el recall por clase real."""
    return {
        "Accuracy general (%)": 100.0 * float(resultado.metricas["accuracy"]),
        **_metricas_por_clase(resultado),
    }


def _fuente_ejecutada(datos: pd.DataFrame) -> dict:
    """Reconstruye los metadatos de la fuente de datos empleada en el experimento."""
    fuente = st.session_state.get("fuente_aplicada", (None, "", ""))
    metadata = st.session_state.get("fuente_metadata", {})
    if metadata.get("identidad") != fuente[2]:
        metadata = {}
    return {
        "archivo": fuente[1],
        "sha256": metadata.get("sha256"),
        "fuente": metadata.get("fuente"),
        "filas": len(datos),
        "columnas": list(datos.columns),
        "filas_archivo": metadata.get("filas"),
        "columnas_archivo": metadata.get("columnas"),
        "lectura": deepcopy(metadata.get("lectura", {})),
        "preparacion_previa": deepcopy(
            st.session_state.get("dataset_preparacion", {})
        ),
    }


def _argumentos_exportacion(
    configuracion: dict,
    opciones: dict,
    fuente: dict,
) -> dict:
    """Construye el diccionario de argumentos de reproducibilidad para exportación."""
    return {
        "csv": fuente["archivo"],
        "target": configuracion["target"],
        "features": list(configuracion["features"]),
        "semilla": configuracion["random_state"],
        "test_size": configuracion["test_size"],
        "validation_size": opciones["validation_size"],
        "estrategia": opciones.get("estrategia", "holdout"),
        "folds": FOLDS_LAB2 if opciones.get("estrategia") == "cv" else None,
        "metrica": opciones["metrica"],
        "algoritmos": list(ALGORITMOS_LAB2),
        "incluir_categoricas": configuracion["incluir_categoricas"],
        "sin_imputar": not configuracion["imputar"],
        "sin_escalar": not configuracion["estandarizar"],
        "estratificar": configuracion["estratificar"],
        "particion_externa": configuracion.get("particion_global") is not None,
        **fuente["lectura"],
        "fuente": fuente["fuente"],
    }
