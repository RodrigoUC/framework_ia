"""Controles pequeños y explícitos para los modelos solicitados en LAB02."""

from __future__ import annotations

import streamlit as st

from ..utils import (
    CONFIGURACIONES_MODELO,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
)

__all__ = ["METRICAS_SELECCION", "NOMBRES_MODELOS", "parametros_individuales"]


def parametros_individuales(algoritmo: str) -> dict:
    """Expone ajustes frecuentes; la configuración estándar la define el dominio."""
    if algoritmo == "NR":
        return {}
    modo = st.radio(
        "Configuración del modelo",
        ["Estándar", "Personalizada"],
        key=f"lab2_modo_{algoritmo}",
        persist_state="session",
        help="Estándar utiliza los valores predeterminados del algoritmo y la semilla indicada.",
    )
    if modo == "Estándar":
        return {}
    prefijo = f"lab2_param_{algoritmo}"
    columnas = st.columns(2)
    resultado = {}
    for indice, parametro in enumerate(CONFIGURACIONES_MODELO[algoritmo].parametros):
        columna = columnas[indice % 2]
        clave = f"{prefijo}_{parametro.nombre}"
        if parametro.tipo == "select":
            formato = None
            if parametro.formato == "distance":
                formato = lambda n: "Euclidiana (p=2)" if n == 2 else "Manhattan (p=1)"
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
