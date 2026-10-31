"""Controles de los agentes; iniciar es explícito y nunca entrena modelos."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from ..utils import (
    AGENTE_CLUSTERING,
    AGENTE_EDA,
    NOMBRES_PROVEEDORES,
    PROVEEDOR_SIN_LLM,
    PROVEEDORES_LLM,
)

SIN_OBJETIVO = "(sin variable objetivo)"


def proveedor_por_defecto() -> str:
    """El primer proveedor con key en el entorno; si no hay, sólo reglas."""
    for codigo, proveedor in PROVEEDORES_LLM.items():
        if proveedor.variable_clave and os.environ.get(proveedor.variable_clave):
            return codigo
    return PROVEEDOR_SIN_LLM


def _controles_llm() -> dict:
    """Proveedor, modelo y API key; deben dibujarse dentro de un formulario."""
    izquierda, derecha = st.columns(2)
    proveedor = izquierda.selectbox(
        "Modelo de lenguaje", list(NOMBRES_PROVEEDORES),
        format_func=NOMBRES_PROVEEDORES.get, key="agentes_proveedor",
    )
    configuracion = PROVEEDORES_LLM.get(proveedor)
    modelo = derecha.text_input(
        "Modelo", value=os.environ.get("AGENTES_MODELO", ""),
        placeholder=(configuracion.modelo_sugerido if configuracion else "") or "nombre del modelo",
        key="agentes_modelo",
        help="Vacío: AGENTES_MODELO o la sugerencia del proveedor.",
    )
    api_key = None
    if configuracion and configuracion.variable_clave:
        variable = configuracion.variable_clave
        api_key = st.text_input(
            "API key", type="password", key="agentes_api_key",
            placeholder=f"{variable} detectada" if os.environ.get(variable) else f"o defina {variable}",
            help="Sólo vive en esta sesión. Preferible: .streamlit/secrets.toml.",
        )
    return {"proveedor": proveedor, "modelo": modelo, "api_key": api_key}


def configurar_agente_clustering() -> dict | None:
    """Usa las variables de la sección Clustering; sólo el botón agrupa."""
    st.session_state.setdefault("agentes_proveedor", proveedor_por_defecto())
    with st.form("form_agente_clustering"):
        st.caption(AGENTE_CLUSTERING.descripcion)
        opciones = _controles_llm()
        iniciar = st.form_submit_button("Iniciar agente", type="primary", icon=":material/smart_toy:")
    return opciones if iniciar else None


def configurar_agente_eda(datos: pd.DataFrame) -> dict | None:
    """Devuelve la configuración sólo cuando se pulsa «Iniciar agente»."""
    st.session_state.setdefault("agentes_proveedor", proveedor_por_defecto())
    with st.form("form_agentes"):
        st.caption(AGENTE_EDA.descripcion)
        columnas = [SIN_OBJETIVO, *map(str, datos.columns)]
        actual = str(st.session_state.get("sup_target", datos.columns[-1]))
        objetivo = st.selectbox(
            "Variable objetivo", columnas,
            index=columnas.index(actual) if actual in columnas else len(columnas) - 1,
            key="agentes_target",
        )
        opciones = _controles_llm()
        iniciar = st.form_submit_button("Iniciar agente", type="primary", icon=":material/smart_toy:")
    if not iniciar:
        return None
    return {
        "target": None if objetivo == SIN_OBJETIVO else objetivo,
        **opciones,
    }
