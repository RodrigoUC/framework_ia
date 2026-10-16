"""Controles nativos aislados de los parámetros de clasificación."""

import math

import streamlit as st

from ..utils.configuracion_regresion import CONFIGURACIONES_REGRESION


def parametros_regresion(algoritmo: str) -> dict | None:
    """Devuelve None si una entrada personalizada no puede ejecutarse."""
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
    for indice, parametro in enumerate(CONFIGURACIONES_REGRESION[algoritmo].parametros):
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
                valor = [float(elemento.strip()) for elemento in texto.split(",")]
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
