"""Navegación lateral, encabezado de ruta, indicadores de calidad y guía de flujo."""

from __future__ import annotations

from html import escape
import re

import pandas as pd
import streamlit as st

from ai_framework.data.eda import EDA
from ai_framework.ui.controllers.unsupervised_controller import (
    _columnas_numericas,
)
from ai_framework.ui.models.view_catalog import (
    APP_NAME,
    RUTAS_VISTA,
    TITULOS_VISTA,
    VISTAS_CLASIFICADORES,
    VISTAS_REGRESION,
    VISTAS_REGRESORES,
)
from ai_framework.ui.models.availability import Disponibilidad, hay_evidencia
from ai_framework.utils.supervised.regression_config import NOMBRES_REGRESORES
from ai_framework.ui.i18n import t, render_selector_idioma, obtener_idioma

def _nombre_regresor(algoritmo: str) -> str:
    lang = obtener_idioma()
    if lang == "en":
        nombres_en = {
            "RLS": "Simple Linear Regression",
            "RLM": "Multiple Linear Regression",
            "Lasso": "Lasso",
            "LassoCV": "LassoCV",
            "Ridge": "Ridge",
            "RidgeCV": "RidgeCV",
            "SVR": "SVR · Support Vector Regressor",
            "DT": "Regression Tree",
            "RF": "Random Forest Regressor",
        }
        return nombres_en.get(algoritmo, NOMBRES_REGRESORES.get(algoritmo, algoritmo))
    return NOMBRES_REGRESORES.get(algoritmo, algoritmo)


def _resumen_calidad(datos: pd.DataFrame) -> None:
    """Muestra indicadores básicos de calidad del dataset activo."""
    resumen = EDA(dataframe=datos).resumen_calidad()
    numericas = _columnas_numericas(datos)
    resumen["outliers_criticos"] = (
        len(
            EDA(dataframe=datos).detectar_outliers(numericas)[
                "filas_con_algun_outlier"
            ]
        )
        if numericas
        else 0
    )
    metricas_en = {
        "filas": "Rows",
        "columnas": "Columns",
        "duplicados": "Duplicates",
        "nulos": "Nulls",
        "porcentaje_nulos": "% Nulls",
        "numericas": "Numerical",
        "categoricas": "Categorical",
        "outliers_criticos": "Critical Outliers",
    }
    lang = obtener_idioma()
    indicadores = list(resumen.items())
    metricas_por_fila = 3
    for inicio in range(0, len(indicadores), metricas_por_fila):
        fila = indicadores[inicio : inicio + metricas_por_fila]
        columnas = st.columns(len(fila))
        for columna, (nombre, valor) in zip(columnas, fila):
            if lang == "en":
                etiqueta = metricas_en.get(nombre, nombre.replace("_", " ").title())
            else:
                etiqueta = nombre.replace("_", " ").title()
            columna.metric(etiqueta, valor)


def _seleccionar_vista(disponibilidades=None) -> str:
    """Renderiza la navegación multinivel y devuelve la vista activa."""
    vista = st.session_state.setdefault("vista_activa", "datos")
    if vista in {"comparacion", "lab2_experimentos"}:
        vista = "lab2_resultados"
        st.session_state["res_subvista_clasificacion"] = "comparativa"
    elif vista == "lab3_comparacion":
        vista = "lab3_resultados"
        st.session_state["res_subvista_regresion"] = "comparativa"
    elif vista in {
        "clasificacion_configuracion",
        "regresion_configuracion",
        "regresion",
    }:
        vista = "configuracion"
    elif vista in {"tsne", "umap"}:
        vista = "kmeans"
    st.session_state["vista_activa"] = vista
    disponibilidades = disponibilidades or {}

    def navegar(destino: str) -> None:
        st.session_state["vista_activa"] = destino

    def boton(
        etiqueta: str,
        destino: str,
        *,
        disabled: bool = False,
        help: str | None = None,
    ) -> None:
        st.button(
            etiqueta,
            key=f"nav_{destino}",
            type="primary" if vista == destino else "secondary",
            disabled=disabled,
            help=help if disabled else None,
            on_click=navegar if not disabled else None,
            args=(destino,) if not disabled else None,
            use_container_width=True,
        )

    with st.sidebar:
        render_selector_idioma()
        st.markdown(
            f'<p class="atlas-nav-label">{t("nav_seccion", "Navegación")}</p>', unsafe_allow_html=True
        )
        boton(t("nav_datos", "Datos y preparación"), "datos")
        st.markdown(
            f'<p class="atlas-nav-label">{t("nav_exploracion", "Exploración de datos")}</p>',
            unsafe_allow_html=True,
        )
        boton(t("nav_eda", "EDA"), "eda")
        boton(t("nav_acp", "ACP"), "acp")
        boton(t("nav_agente_ia", "Agente IA"), "eda_agente")

        st.markdown(
            f'<p class="atlas-nav-label">{t("nav_analisis", "Análisis y modelos")}</p>',
            unsafe_allow_html=True,
        )
        boton(t("nav_configuracion", "Configuración"), "configuracion")
        with st.expander(
            t("nav_clustering", "Clustering"),
            expanded=vista in {"kmeans", "kmedoids", "hac", "clustering_agente"},
        ):
            boton("K-Means", "kmeans")
            boton("K-Medoids", "kmedoids")
            boton("HAC", "hac")
            boton(t("nav_agente_ia", "Agente IA"), "clustering_agente")

        with st.expander(
            t("nav_clasificacion", "Clasificación"),
            expanded=vista
            in {*VISTAS_CLASIFICADORES, "lab2_experimentos", "lab2_benchmark"},
        ):
            disponible = disponibilidades.get("clasificacion")
            bloqueada = disponible is not None and not disponible.lista
            motivo = " ".join(disponible.faltantes) if bloqueada else None
            st.markdown(
                f'<p class="atlas-family">{t("nav_modelos_disponibles", "Modelos disponibles")}</p>',
                unsafe_allow_html=True,
            )
            boton("KNN", "clasificacion_knn", disabled=bloqueada, help=motivo)
            boton(
                t("nav_dt", "Árbol de decisión"),
                "clasificacion_dt",
                disabled=bloqueada,
                help=motivo,
            )
            boton(
                t("nav_rf", "Random Forest"),
                "clasificacion",
                disabled=bloqueada,
                help=motivo,
            )
            boton(
                "XGBoost",
                "clasificacion_xgboost",
                disabled=bloqueada,
                help=motivo,
            )
            boton(
                "AdaBoost",
                "clasificacion_adaboost",
                disabled=bloqueada,
                help=motivo,
            )
            boton(
                "Naive Bayes",
                "clasificacion_nb",
                disabled=bloqueada,
                help=motivo,
            )
            boton(
                t("nav_benchmark", "Evaluación Comparativa"),
                "lab2_benchmark",
                disabled=bloqueada,
                help=motivo,
            )

        with st.expander(t("nav_regresion", "Regresión"), expanded=vista in VISTAS_REGRESION):
            disponible = disponibilidades.get("regresion")
            bloqueada = disponible is not None and not disponible.lista
            motivo = " ".join(disponible.faltantes) if bloqueada else None
            for destino, algoritmo in VISTAS_REGRESORES.items():
                bloqueo_modelo = bloqueada and not (
                    algoritmo == "RLS"
                    and disponibilidades.get("RLS", disponible).lista
                )
                boton(
                    _nombre_regresor(algoritmo),
                    destino,
                    disabled=bloqueo_modelo,
                    help=motivo,
                )
            boton(
                t("nav_benchmark", "Evaluación Comparativa"),
                "lab3_benchmark",
                disabled=bloqueada
                and not disponibilidades.get("RLS", disponible).lista,
                help=motivo,
            )

        st.markdown(
            f'<p class="atlas-nav-label">{t("nav_resultados", "Resultados")}</p>', unsafe_allow_html=True
        )
        for familia, destino, etiqueta_k, etiqueta_d in (
            ("clasificacion", "lab2_resultados", "nav_resultados_clasificacion", "Resultados de clasificación"),
            ("regresion", "lab3_resultados", "nav_resultados_regresion", "Resultados de regresión"),
        ):
            evidencia = hay_evidencia(st.session_state, familia)
            boton(
                t(etiqueta_k, etiqueta_d),
                destino,
                disabled=not evidencia,
                help=t("nav_resultados_help", "Ejecute primero un modelo o evaluación comparativa."),
            )
    return st.session_state["vista_activa"]


def _titulo_vista(vista: str) -> str:
    lang = obtener_idioma()
    if vista == "configuracion":
        return t("nav_configuracion", "Configuración")
    if vista in VISTAS_REGRESION:
        titulos_regresion = {
            **{
                ruta: _nombre_regresor(alg)
                for ruta, alg in VISTAS_REGRESORES.items()
            },
            "regresion": t("nav_configuracion", "Configuración"),
            "regresion_configuracion": t("nav_configuracion", "Configuración"),
            "lab3_benchmark": t("nav_benchmark_completo", "Evaluación Comparativa de Modelos"),
            "lab3_comparacion": t("res_eval_comp", "Comparar modelos entrenados"),
            "lab3_resultados": t("nav_resultados_regresion", "Resultados de regresión"),
        }
        return titulos_regresion.get(vista, TITULOS_VISTA.get(vista, vista))
    if lang == "en":
        titulos_en = {
            "datos": "Data & Preparation",
            "eda": "Data Exploration",
            "acp": "PCA",
            "kmeans": "K-Means",
            "kmedoids": "K-Medoids",
            "hac": "Hierarchical Clustering",
            "eda_agente": "EDA Agent",
            "clustering_agente": "Clustering Agent",
            "clasificacion": "Random Forest",
            "clasificacion_knn": "KNN",
            "clasificacion_dt": "Decision Tree",
            "clasificacion_xgboost": "XGBoost",
            "clasificacion_adaboost": "AdaBoost",
            "clasificacion_nb": "Naive Bayes",
            "clasificacion_configuracion": "Settings",
            "lab2_experimentos": "Comparative Evaluation",
            "lab2_benchmark": "Model Comparative Evaluation",
            "lab2_resultados": "Classification Results",
        }
        return titulos_en.get(vista, TITULOS_VISTA.get(vista, vista))
    return TITULOS_VISTA.get(vista, vista)


def _ruta_vista(vista: str, titulo: str) -> str:
    lang = obtener_idioma()
    if vista == "configuracion":
        return t("nav_configuracion", "Configuración")
    if vista in VISTAS_REGRESION:
        pref = "Regression" if lang == "en" else "Regresión"
        return f"{pref} / {titulo}"
    if lang == "en":
        rutas_en = {
            "datos": "Data / Preparation",
            "eda": "Exploration / EDA",
            "acp": "Exploration / PCA",
            "kmeans": "Clustering / K-Means",
            "kmedoids": "Clustering / K-Medoids",
            "hac": "Clustering / Hierarchical",
            "eda_agente": "Exploration / AI Agent",
            "clustering_agente": "Clustering / AI Agent",
            "clasificacion": "Classification / Random Forest",
            "clasificacion_knn": "Classification / KNN",
            "clasificacion_dt": "Classification / Decision Tree",
            "clasificacion_xgboost": "Classification / XGBoost",
            "clasificacion_adaboost": "Classification / AdaBoost",
            "clasificacion_nb": "Classification / Naive Bayes",
            "clasificacion_configuracion": "Settings",
            "lab2_experimentos": "Classification / Benchmark",
            "lab2_benchmark": "Classification / Benchmark",
            "lab2_resultados": "Results / Classification",
        }
        return rutas_en.get(vista, f"Analysis / {titulo}")
    return RUTAS_VISTA.get(vista, f"Análisis / {titulo}")


def _render_encabezado_atlas(
    datos: pd.DataFrame, etiqueta: str, vista: str
) -> None:
    """Muestra el encabezado del dataset activo y el resumen de calidad en vistas exploratorias."""
    titulo = _titulo_vista(vista)
    ruta = _ruta_vista(vista, titulo)
    nombre_dataset = escape(etiqueta)
    lbl_dataset = t("dataset_activo", "Dataset activo")
    st.markdown(
        f"""
        <section class="atlas-header">
          <div class="atlas-breadcrumb">{APP_NAME} / {ruta}</div>
          <h1 class="atlas-title">{titulo}</h1>
          <p class="atlas-subtitle">{lbl_dataset}: <strong>{nombre_dataset}</strong></p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if vista in {"datos", "eda"}:
        _resumen_calidad(datos)


def _traducir_mensaje_faltante(mensaje: str) -> str:
    if obtener_idioma() != "en":
        return mensaje
    msg = mensaje
    prefijo = ""
    if msg.startswith("Revise Configuración de clasificación:"):
        prefijo = "Check Classification Configuration:"
        msg = msg[len("Revise Configuración de clasificación:"):].strip()
    elif msg.startswith("Revise Configuración de regresión:"):
        prefijo = "Check Regression Configuration:"
        msg = msg[len("Revise Configuración de regresión:"):].strip()

    m_cv = re.match(r"CV necesita al menos (\d+) filas por clase en entrenamiento\. Use holdout o aporte más ejemplos\.", msg)
    if m_cv:
        msg = f"CV requires at least {m_cv.group(1)} rows per class in training. Use holdout or provide more examples."
    elif "Se necesitan una columna objetivo y al menos una predictora" in msg:
        msg = "A target column and at least one predictor are required. Check the source in Data and preparation."
    elif "Prueba necesita al menos dos filas para R²" in msg:
        msg = "Test requires at least two rows for R². Increase test size or provide more data."
    elif msg == "Entrenamiento necesita al menos dos filas.":
        msg = "Training requires at least two rows."
    else:
        m_alg = re.match(r"(.+) necesita al menos cinco filas de entrenamiento para su validación interna\.", msg)
        if m_alg:
            msg = f"{m_alg.group(1)} requires at least five training rows for its internal validation."

    if prefijo:
        return f"{prefijo} {msg}"
    return msg


def mostrar_faltantes(disponibilidad: Disponibilidad) -> None:
    """Presenta advertencias para cada prerrequisito faltante en la vista."""
    for mensaje in disponibilidad.faltantes:
        st.warning(_traducir_mensaje_faltante(mensaje))


def ir_a(destino: str) -> None:
    """Actualiza la vista activa en la sesión de Streamlit."""
    st.session_state["vista_activa"] = destino


def mostrar_guia(
    vista, familia=None, disponibilidad=None, configuracion=None
) -> None:
    """Muestra la orientación contextual de la etapa actual y acciones siguientes."""
    del configuracion
    if familia is None:
        if vista == "configuracion":
            st.markdown("<!-- Etapa 2 · Configuración -->", unsafe_allow_html=True)
            return
        if vista == "datos":
            st.markdown("<!-- Etapa 1 · Fuente activa -->", unsafe_allow_html=True)
            st.button(
                t("configurar_tarea", "Configurar tarea"),
                key="flujo_configurar",
                on_click=ir_a,
                args=("configuracion",),
            )
        return
    nombre = "regresión" if familia == "regresion" else "clasificación"
    configurando = vista in (
        "configuracion",
        "regresion",
        "regresion_configuracion",
        "clasificacion_configuracion",
    )
    consultando = vista in (
        "lab2_resultados",
        "lab2_experimentos",
        "lab3_resultados",
        "lab3_comparacion",
    )
    etapa = (
        "2 · Configuración"
        if configurando
        else "4 · Resultados"
        if consultando
        else "3 · Ejecución"
    )
    st.markdown(f"<!-- Etapa {etapa} de {nombre} -->", unsafe_allow_html=True)

    if disponibilidad is not None and not disponibilidad.lista:
        if configurando or consultando:
            mostrar_faltantes(disponibilidad)

    if configurando:
        if disponibilidad is not None:
            destino = "regresion_ridge" if familia == "regresion" else "clasificacion"
            st.button(
                t("flujo_continuar_modelo", "Continuar a un modelo"),
                key="flujo_continuar",
                disabled=not disponibilidad.lista,
                on_click=ir_a,
                args=(destino,),
            )
    else:
        # En ejecución y resultados, botones discretos de soporte en columnas compactas
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            st.button(
                t("flujo_revisar_config", "Revisar configuración"),
                key="flujo_corregir",
                on_click=ir_a,
                args=("configuracion",),
            )
        if not consultando and hay_evidencia(st.session_state, familia):
            destino_res = (
                "lab3_resultados" if familia == "regresion" else "lab2_resultados"
            )
            with col2:
                st.button(
                    t("flujo_revisar_resultados", "Revisar resultados guardados"),
                    key="flujo_resultados",
                    on_click=ir_a,
                    args=(destino_res,),
                )
