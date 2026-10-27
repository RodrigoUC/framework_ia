"""Controles reutilizables de entrada e hiperparámetros para la interfaz."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from framework_ia.utils import (
    CONFIGURACIONES_MODELO,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
)
from framework_ia.utils.configuracion_regresion import CONFIGURACIONES_REGRESION


def _seleccionar_entero(
    etiqueta: str,
    minimo: int,
    maximo: int,
    predeterminado: int,
    *,
    key: str,
) -> int:
    """Renderiza un control deslizante entero o un texto fijo si el rango es unitario."""
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


def _mostrar_dataframe(data: pd.DataFrame, *, alt: str, **opciones) -> None:
    """Muestra una descripción accesible junto a la tabla de datos."""
    st.caption(alt)
    st.dataframe(data, **opciones)


def parametros_individuales(algoritmo: str) -> dict:
    """Renderiza los controles de hiperparámetros para un modelo de clasificación."""
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
        "Configuración del modelo",
        ["Estándar", "Personalizada"],
        key=f"lab2_modo_{algoritmo}",
        persist_state="session",
        help="Estándar utiliza los valores predeterminados del algoritmo y la semilla indicada.",
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


def parametros_regresion(algoritmo: str) -> dict | None:
    """Renderiza los controles de hiperparámetros para un modelo de regresión."""
    modo = st.radio(
        "Configuración del modelo",
        ["Estándar", "Personalizada"],
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
