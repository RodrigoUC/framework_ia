"""Componentes reutilizables y unificados para las vistas de Resultados (Clasificación y Regresión)."""

from __future__ import annotations

from collections.abc import Callable
import pandas as pd
import streamlit as st

from framework_ia.ui.vistas.tema import _obtener_pestanas, render_tabla_clave_valor


def render_panel_modelos_individuales(
    df_metricas: pd.DataFrame,
    modelos_params: list[tuple[str, dict]],
    *,
    titulo: str = "Modelos entrenados individualmente",
) -> None:
    """Renderiza de forma unificada la tabla de modelos individuales y el visor de hiperparámetros."""
    st.markdown(f"#### {titulo}")
    st.dataframe(df_metricas, hide_index=True, width="stretch")

    if modelos_params:
        with st.expander("Ver hiperparámetros de los modelos"):
            nombres = [nombre for nombre, _ in modelos_params]
            if len(nombres) == 1:
                render_tabla_clave_valor(
                    modelos_params[0][1],
                    columna_clave="Hiperparámetro",
                    columna_valor="Valor configurado",
                )
            else:
                tabs = _obtener_pestanas(nombres)
                for tab, (nombre, params) in zip(tabs, modelos_params):
                    with tab:
                        render_tabla_clave_valor(
                            params,
                            columna_clave="Hiperparámetro",
                            columna_valor="Valor configurado",
                        )


def render_layout_resultados(
    *,
    hay_individuales: bool,
    hay_comparativa: bool,
    render_individuales_fn: Callable[[], None],
    render_comparativa_fn: Callable[[], None],
    mensaje_sin_datos: str = "Todavía no hay modelos entrenados ni evaluaciones guardadas en este contexto.",
) -> None:
    """Organiza simétricamente la página de resultados en dos columnas verticales cuando coexisten datos."""
    if not hay_individuales and not hay_comparativa:
        st.info(mensaje_sin_datos)
        return

    if hay_individuales and hay_comparativa:
        col_ind, col_exp = st.columns(2, gap="large")
        with col_ind:
            render_individuales_fn()
        with col_exp:
            render_comparativa_fn()
    elif hay_individuales:
        render_individuales_fn()
    elif hay_comparativa:
        render_comparativa_fn()
