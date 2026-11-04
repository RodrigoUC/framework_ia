"""Reusable input controls and hyperparameter widgets for the UI."""

from __future__ import annotations
import json

import math

import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.utils import (
    CONFIGURACIONES_MODELO,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
)
from ai_framework.utils.supervised.regression_config import CONFIGURACIONES_REGRESION


def _select_integer(
    etiqueta: str,
    minimo: int,
    maximo: int,
    predeterminado: int,
    *,
    key: str,
) -> int:
    """Render integer slider or fixed text if range has a single value."""
    if minimo == maximo:
        st.caption(f"{etiqueta}: {minimo}")
        return minimo
    return st.slider(
        etiqueta,
        minimo,
        maximo,
        min(max(predeterminado, minimo), maximo),
        key=key,
    )


def _sanitize_dataframe_arrow(data: pd.DataFrame) -> pd.DataFrame:
    """Avoid PyArrow serialization failures by converting dict/list columns to JSON/string."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data
    modificado = False
    vista = data
    for col in data.columns:
        serie = data[col]
        if serie.dtype == "object":
            if any(isinstance(v, (dict, list, tuple, set)) for v in serie.dropna().head(20)):
                if not modificado:
                    vista = data.copy()
                    modificado = True
                vista[col] = vista[col].map(
                    lambda x: json.dumps(x, ensure_ascii=False, default=str)
                    if isinstance(x, (dict, list, tuple, set))
                    else (str(x) if x is not None else "")
                )
    return vista


def _display_dataframe(data: pd.DataFrame, *, alt: str, **opciones) -> None:
    """Display accessible description alongside data table."""
    st.caption(alt)
    st.dataframe(_sanitize_dataframe_arrow(data), **opciones)


def individual_parameters(algoritmo: str) -> dict:
    """Render hyperparameter controls for a classification model."""
    if algoritmo == "NR":
        return {}
    criterio_rf = None
    if algoritmo == "RF":
        criterio_rf = st.selectbox(
            "Criterio de división",
            ["gini", "entropy"],
            key="lab2_param_RF_criterion",
            persist_state="session",
            help="Cada criterio conserva su última ejecución exitosa por separado.",
        )
    modo = st.radio(
        t("sup_configuracion_modelo", "Configuración del modelo"),
        ["Estándar", "Personalizada"],
        format_func=lambda x: {"Estándar": t("modo_estandar", "Estándar"), "Personalizada": t("modo_personalizada", "Personalizada")}.get(x, x),
        key=f"lab2_modo_{algoritmo}",
        persist_state="session",
        help=t("modo_help", "Estándar utiliza los valores predeterminados del algoritmo y la semilla indicada."),
    )
    if modo == "Estándar":
        return {"criterion": criterio_rf} if criterio_rf is not None else {}
    prefijo = f"lab2_param_{algoritmo}"
    columnas = st.columns(2)
    resultado = {"criterion": criterio_rf} if criterio_rf is not None else {}
    for indice, parametro in enumerate(
        CONFIGURACIONES_MODELO[algoritmo].parametros
    ):
        columna = columnas[indice % 2]
        clave = f"{prefijo}_{parametro.nombre}"
        if parametro.tipo == "select":
            formato = None
            if parametro.formato == "distance":
                formato = (
                    lambda n: "Euclidiana (p=2)"
                    if n == 2
                    else "Manhattan (p=1)"
                )
            valor = columna.selectbox(
                parametro.etiqueta,
                list(parametro.opciones),
                key=clave,
                persist_state="session",
                **({"format_func": formato} if formato is not None else {}),
            )
        else:
            valor = columna.number_input(
                parametro.etiqueta,
                min_value=parametro.minimo,
                max_value=parametro.maximo,
                value=parametro.valor,
                step=parametro.paso,
                key=clave,
                persist_state="session",
            )
            if parametro.tipo == "depth":
                valor = valor or None
        resultado[parametro.nombre] = valor
    return resultado


def regression_parameters(algoritmo: str) -> dict | None:
    """Render hyperparameter controls for a regression model."""
    modo = st.radio(
        t("sup_configuracion_modelo", "Configuración del modelo"),
        ["Estándar", "Personalizada"],
        format_func=lambda x: {"Estándar": t("modo_estandar", "Estándar"), "Personalizada": t("modo_personalizada", "Personalizada")}.get(x, x),
        key=f"lab3_modo_{algoritmo}",
        persist_state="session",
    )
    if algoritmo in ("LassoCV", "RidgeCV"):
        st.caption(
            "Validación interna fija: KFold de 5 pliegues, barajado y semilla compartida. Selecciona alpha solo con train; no reajusta la preparación dentro de cada pliegue interno."
        )
    if modo == "Estándar":
        return {}
    resultado = {}
    columnas = st.columns(2)
    for indice, parametro in enumerate(
        CONFIGURACIONES_REGRESION[algoritmo].parametros
    ):
        columna = columnas[indice % 2]
        clave = f"lab3_param_{algoritmo}_{parametro.nombre}"
        if parametro.tipo == "select":
            valor = columna.selectbox(
                parametro.etiqueta,
                parametro.opciones,
                key=clave,
                persist_state="session",
            )
        elif parametro.tipo == "alphas":
            texto = columna.text_input(
                parametro.etiqueta,
                value=", ".join(map(str, parametro.valor)),
                key=clave,
                persist_state="session",
            )
            try:
                valor = [
                    float(elemento.strip()) for elemento in texto.split(",")
                ]
                if not valor or any(
                    not math.isfinite(alpha) or alpha <= 0 for alpha in valor
                ):
                    raise ValueError
            except ValueError:
                st.error(
                    "Introduzca valores de alpha positivos y finitos separados por comas."
                )
                return None
        else:
            valor = columna.number_input(
                parametro.etiqueta,
                min_value=parametro.minimo,
                max_value=parametro.maximo,
                value=parametro.valor,
                step=parametro.paso,
                key=clave,
                persist_state="session",
            )
            if parametro.tipo == "depth":
                valor = valor or None
        resultado[parametro.nombre] = valor
    return resultado

# Backward compatibility aliases
_seleccionar_entero = _select_integer
_sanear_dataframe_arrow = _sanitize_dataframe_arrow
_mostrar_dataframe = _display_dataframe
parametros_individuales = individual_parameters
parametros_regresion = regression_parameters
