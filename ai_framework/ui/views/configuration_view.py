"""Vistas de configuración para flujos supervisados y no supervisados."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.data.eda import EDA
from ai_framework.ui.controllers.sources_controller import _usar_particion
from ai_framework.ui.controllers.unsupervised_controller import (
    _columnas_numericas,
)
from ai_framework.ui.models.view_catalog import TAREAS
from ai_framework.ui.models.classification_state import (
    configuracion_clasificacion_actual,
)
from ai_framework.ui.models.regression_state import (
    configuracion_regresion_actual,
)
from ai_framework.ui.models.session_state import inicializar_configuracion


def configurar_supervisado(datos: pd.DataFrame) -> None:
    """Renderiza los controles de preparación y protocolo para tareas supervisadas."""
    tarea = st.radio(
        t("cfg_tarea", "Tarea"),
        list(TAREAS),
        index=None,
        format_func=lambda x: t(f"tarea_{x}", TAREAS.get(x, x)),
        key="sup_tarea",
        persist_state="session",
        horizontal=True,
    )
    if len(datos.columns) < 2:
        st.warning("Se necesita un objetivo y al menos una variable predictora.")
        return
    if tarea is None:
        st.info(
            "Seleccione Clasificación para clases o Regresión para cantidades."
        )
        return
    target = st.selectbox(
        t("cfg_target", "Variable objetivo (target)"),
        datos.columns.tolist(),
        key="sup_target",
        persist_state="session",
    )
    st.multiselect(
        t("cfg_features", "Variables predictoras"),
        [col for col in datos if col != target],
        key="sup_features",
        persist_state="session",
    )
    with st.expander(t("cfg_distribucion_target", "Distribución del objetivo")): 
        serie = datos[target]
        st.caption(f"Filas: {len(serie)} · Nulos: {int(serie.isna().sum())}")
        if tarea == "clasificacion":
            resumen, _ = EDA(dataframe=datos).balance_objetivo(target)
            if resumen.empty:
                resumen = (
                    serie.value_counts(dropna=False)
                    .rename("Cantidad")
                    .to_frame()
                )
        else:
            if not pd.api.types.is_numeric_dtype(serie):
                st.warning("La variable objetivo debe ser numérica para regresión.")
                resumen = serie.describe().astype(str).rename("Valor").to_frame()
            else:
                resumen = serie.describe().rename("Valor").to_frame()
        for col in resumen.columns:
            if resumen[col].dtype == object:
                resumen[col] = resumen[col].astype(str)
        st.dataframe(resumen, width="stretch")
    if tarea == "regresion":
        simples = datos[st.session_state["sup_features"]].select_dtypes(
            include="number"
        )
        st.selectbox(
            t("cfg_feature_simple", "Predictor de regresión lineal simple"),
            simples.columns.tolist(),
            key="sup_feature_simple",
            persist_state="session",
        )
    st.markdown(f"#### {t('cfg_particion_reproducibilidad', 'Partición y reproducibilidad')}")
    if st.session_state.get("sup_aviso_particion"):
        st.info(st.session_state["sup_aviso_particion"])
    externa = bool(
        st.session_state.get("particion_global") is not None
        and st.session_state.get("sup_usar_particion_global")
    )
    st.slider(
        t("cfg_test_size", "Tamaño de prueba (%)"),
        10,
        60,
        step=5,
        key="sup_test_size",
        persist_state="session",
        disabled=externa,
    )
    st.number_input(
        t("cfg_random_state", "Semilla"),
        min_value=0,
        max_value=10_000,
        step=1,
        key="sup_random_state",
        persist_state="session",
        disabled=externa,
    )
    if tarea == "clasificacion":
        st.checkbox(
            t("cfg_estratificar", "Estratificar por target"),
            key="sup_strat",
            persist_state="session",
            disabled=externa,
        )
    if st.session_state.get("particion_global") is not None:
        st.checkbox(
            t("cfg_usar_particion", "Usar partición calculada"),
            key="sup_usar_particion_global",
            persist_state="session",
            on_change=_usar_particion,
        )
    st.markdown(f"#### {t('cfg_prep_entrenamiento', 'Preparación aprendida solo con entrenamiento')}")
    for etiqueta, clave in (
        (t("cfg_codificar_cat", "Codificar categóricas"), "incluir_categoricas"),
        (t("cfg_imputar", "Imputar nulos de predictores"), "imputar"),
        (t("cfg_escalar", "Escalar predictores numéricos"), "escalar"),
    ):
        st.checkbox(etiqueta, key=f"sup_{clave}", persist_state="session")


def configurar_clasificacion(
    datos: pd.DataFrame, mostrar_figura: Callable | None = None
) -> dict | None:
    """Muestra la configuración supervisada compartida y retorna la configuración de clasificación."""
    del mostrar_figura
    inicializar_configuracion(datos, st.session_state)
    configurar_supervisado(datos)
    return configuracion_clasificacion_actual(datos, st.session_state)


def configurar_regresion(datos: pd.DataFrame) -> dict | None:
    """Muestra la configuración supervisada compartida y retorna la configuración de regresión."""
    inicializar_configuracion(datos, st.session_state)
    configurar_supervisado(datos)
    return configuracion_regresion_actual(datos, st.session_state)


def _configurar_modelos(datos: pd.DataFrame) -> dict:
    """Recopila una configuración común para todos los modelos no supervisados."""
    st.markdown(f"### {t('cfg_analisis_titulo', 'Configuración del análisis')}")
    incluir_categoricas = st.checkbox(
        "Codificar variables categóricas",
        value=False,
        key="modelo_categoricas",
    )
    if incluir_categoricas:
        opciones = datos.columns.tolist()
        predeterminadas = datos.columns.tolist()
    else:
        opciones = _columnas_numericas(datos)
        predeterminadas = opciones
    guardadas = st.session_state.get("modelo_features", [])
    if any(columna not in opciones for columna in guardadas):
        del st.session_state["modelo_features"]
    features = st.multiselect(
        t("cfg_variables_analisis", "Variables de análisis"),
        options=opciones,
        default=predeterminadas,
        key="modelo_features",
    )
    estandarizar = st.checkbox(
        t("cfg_estandarizar_modelado", "Estandarizar para modelado"),
        value=True,
        key="modelo_estandarizar",
    )
    maximo = min(len(datos), 3000)
    if maximo > 20:
        filas = st.slider(
            t("cfg_max_filas_modelo", "Máximo de filas por modelo"),
            min_value=20,
            max_value=maximo,
            value=maximo,
            step=max(1, maximo // 20),
            key="modelo_filas",
        )
    else:
        filas = maximo
    return {
        "features": features,
        "incluir_categoricas": incluir_categoricas,
        "estandarizar": estandarizar,
        "filas": filas,
    }


def _render_regresion() -> None:
    """Orienta hacia las vistas de configuración y modelos de regresión."""
    st.info(
        "Abra Configuración para elegir target y predictores; "
        "después seleccione un modelo o Experimentos en la navegación."
    )


def _render_configuracion_requerida(*, sin_features: bool = False) -> None:
    """Ofrece un acceso directo cuando falta configurar variables de clasificación."""
    st.info(
        "Seleccione al menos una variable predictora en Configuración."
        if sin_features
        else "La clasificación necesita al menos una variable objetivo y una "
        "variable predictora. Revise la configuración compartida antes de continuar."
    )
    if st.button(
        "Abrir configuración de clasificación",
        key="abrir_configuracion_clasificacion",
    ):
        st.session_state["vista_activa"] = "clasificacion_configuracion"
        st.rerun()
