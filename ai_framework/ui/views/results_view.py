"""Componentes reutilizables y unificados para las vistas de Resultados (Clasificación y Regresión)."""

from __future__ import annotations

from collections.abc import Callable
import pandas as pd
import streamlit as st

from ai_framework.ui.i18n import t
from ai_framework.ui.views.theme import _obtener_pestanas, render_tabla_clave_valor


def render_panel_modelos_individuales(
    df_metricas: pd.DataFrame,
    modelos_params: list[tuple[str, dict]],
    *,
    titulo: str | None = None,
) -> None:
    """Renderiza de forma unificada la tabla de modelos individuales y el visor de hiperparámetros."""
    titulo_texto = titulo if titulo is not None else t("res_modelos_ind", "Modelos entrenados individualmente")
    st.markdown(f"#### {titulo_texto}")
    st.dataframe(df_metricas, hide_index=True, width="stretch")

    if modelos_params:
        with st.expander(t("res_ver_hiperparametros", "Ver hiperparámetros de los modelos")):
            nombres = [nombre for nombre, _ in modelos_params]
            col_param = t("res_parametro", "Hiperparámetro")
            col_val = t("res_valor", "Valor configurado")
            if len(nombres) == 1:
                render_tabla_clave_valor(
                    modelos_params[0][1],
                    columna_clave=col_param,
                    columna_valor=col_val,
                )
            else:
                tabs = _obtener_pestanas(nombres)
                for tab, (nombre, params) in zip(tabs, modelos_params):
                    with tab:
                        render_tabla_clave_valor(
                            params,
                            columna_clave=col_param,
                            columna_valor=col_val,
                        )


def render_layout_resultados(
    *,
    hay_individuales: bool,
    hay_comparativa: bool,
    render_individuales_fn: Callable[[], None],
    render_comparativa_fn: Callable[[], None],
    mensaje_sin_datos: str | None = None,
    clave_contexto: str = "general",
) -> None:
    """Organiza la página de resultados con dos botones para conmutar entre modelos individuales y evaluación comparativa."""
    if not hay_individuales and not hay_comparativa:
        msg = mensaje_sin_datos or t("res_sin_datos", "Todavía no hay modelos entrenados ni evaluaciones guardadas en este contexto.")
        st.info(msg)
        return

    clave_estado = f"res_subvista_{clave_contexto}"
    
    if clave_estado not in st.session_state:
        st.session_state[clave_estado] = "individuales" if hay_individuales else "comparativa"

    activo = st.session_state[clave_estado]
    if activo == "individuales" and not hay_individuales and hay_comparativa:
        activo = "comparativa"
        st.session_state[clave_estado] = "comparativa"
    elif activo == "comparativa" and not hay_comparativa and hay_individuales:
        activo = "individuales"
        st.session_state[clave_estado] = "individuales"

    # Mostrar dos botones para alternar entre una u otra vista
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button(
            t("res_modelos_ind", "Modelos individuales"),
            key=f"{clave_contexto}_btn_subvista_ind",
            type="primary" if activo == "individuales" else "secondary",
            use_container_width=True,
            disabled=not hay_individuales,
            help=None if hay_individuales else t("res_sin_modelos_ind", "No se han entrenado modelos individuales."),
        ):
            st.session_state[clave_estado] = "individuales"
            st.rerun()
    with col2:
        if st.button(
            t("res_eval_comp", "Evaluación comparativa"),
            key=f"{clave_contexto}_btn_subvista_comp",
            type="primary" if activo == "comparativa" else "secondary",
            use_container_width=True,
            disabled=not hay_comparativa,
            help=None if hay_comparativa else t("res_sin_benchmark", "No se ha ejecutado la evaluación comparativa."),
        ):
            st.session_state[clave_estado] = "comparativa"
            st.rerun()

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Mostrar ÚNICAMENTE la subvista activa a la vez
    if activo == "individuales":
        render_individuales_fn()
    else:
        render_comparativa_fn()
