"""Navegación lateral, encabezado de ruta, indicadores de calidad y guía de flujo."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from framework_ia.datos.eda import EDA
from framework_ia.ui.controladores.controlador_no_supervisado import (
    _columnas_numericas,
)
from framework_ia.ui.modelos.catalogo_vistas import (
    APP_NAME,
    RUTAS_VISTA,
    TITULOS_VISTA,
    VISTAS_CLASIFICADORES,
    VISTAS_REGRESION,
    VISTAS_REGRESORES,
)
from framework_ia.ui.modelos.disponibilidad import Disponibilidad, hay_evidencia
from framework_ia.utils.configuracion_regresion import NOMBRES_REGRESORES


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
    indicadores = list(resumen.items())
    metricas_por_fila = 3
    for inicio in range(0, len(indicadores), metricas_por_fila):
        fila = indicadores[inicio : inicio + metricas_por_fila]
        columnas = st.columns(len(fila))
        for columna, (nombre, valor) in zip(columnas, fila):
            etiqueta = nombre.replace("_", " ").title()
            columna.metric(etiqueta, valor)


def _seleccionar_vista(disponibilidades=None) -> str:
    """Renderiza la navegación multinivel y devuelve la vista activa."""
    vista = st.session_state.setdefault("vista_activa", "datos")
    if vista in {"comparacion", "lab2_experimentos"}:
        vista = "lab2_resultados"
    elif vista == "lab3_comparacion":
        vista = "lab3_resultados"
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
        )

    with st.sidebar:
        st.markdown(
            '<p class="atlas-nav-label">Navegación</p>', unsafe_allow_html=True
        )
        boton("Datos y preparación", "datos")
        st.markdown(
            '<p class="atlas-nav-label">Exploración de datos</p>',
            unsafe_allow_html=True,
        )
        boton("EDA", "eda")
        boton("ACP", "acp")

        st.markdown(
            '<p class="atlas-nav-label">Análisis y modelos</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="atlas-nav-label">Configuración</p>',
            unsafe_allow_html=True,
        )
        boton("Configuración", "configuracion")
        with st.expander(
            "Clustering", expanded=vista in {"kmeans", "kmedoids", "hac"}
        ):
            boton("K-Means", "kmeans")
            boton("K-Medoids", "kmedoids")
            boton("HAC", "hac")

        with st.expander(
            "Clasificación",
            expanded=vista
            in {*VISTAS_CLASIFICADORES, "lab2_experimentos", "lab2_benchmark"},
        ):
            disponible = disponibilidades.get("clasificacion")
            bloqueada = disponible is not None and not disponible.lista
            motivo = " ".join(disponible.faltantes) if bloqueada else None
            st.markdown(
                '<p class="atlas-family">Modelos disponibles</p>',
                unsafe_allow_html=True,
            )
            boton("KNN", "clasificacion_knn", disabled=bloqueada, help=motivo)
            boton(
                "Árbol de decisión",
                "clasificacion_dt",
                disabled=bloqueada,
                help=motivo,
            )
            boton(
                "Random Forest",
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
                "Experimentos Lab02",
                "lab2_benchmark",
                disabled=bloqueada,
                help=motivo,
            )

        with st.expander("Regresión", expanded=vista in VISTAS_REGRESION):
            disponible = disponibilidades.get("regresion")
            bloqueada = disponible is not None and not disponible.lista
            motivo = " ".join(disponible.faltantes) if bloqueada else None
            for destino, algoritmo in VISTAS_REGRESORES.items():
                bloqueo_modelo = bloqueada and not (
                    algoritmo == "RLS"
                    and disponibilidades.get("RLS", disponible).lista
                )
                boton(
                    NOMBRES_REGRESORES[algoritmo],
                    destino,
                    disabled=bloqueo_modelo,
                    help=motivo,
                )
            boton(
                "Experimentos Lab03",
                "lab3_benchmark",
                disabled=bloqueada
                and not disponibilidades.get("RLS", disponible).lista,
                help=motivo,
            )

        st.markdown(
            '<p class="atlas-nav-label">Resultados</p>', unsafe_allow_html=True
        )
        for familia, destino, etiqueta in (
            ("clasificacion", "lab2_resultados", "Resultados de clasificación"),
            ("regresion", "lab3_resultados", "Resultados de regresión"),
        ):
            evidencia = hay_evidencia(st.session_state, familia)
            boton(
                etiqueta,
                destino,
                disabled=not evidencia,
                help="Ejecute primero un modelo o experimento del contexto actual.",
            )
    return st.session_state["vista_activa"]


def _render_encabezado_atlas(
    datos: pd.DataFrame, etiqueta: str, vista: str
) -> None:
    """Muestra el encabezado del dataset activo y el resumen de calidad en vistas exploratorias."""
    titulos_regresion = {
        **{
            ruta: NOMBRES_REGRESORES[alg]
            for ruta, alg in VISTAS_REGRESORES.items()
        },
        "regresion": "Configuración",
        "regresion_configuracion": "Configuración",
        "lab3_benchmark": "Experimentos Lab03",
        "lab3_comparacion": "Comparar regresores entrenados",
        "lab3_resultados": "Resultados de regresión",
    }
    titulo = (
        "Configuración"
        if vista == "configuracion"
        else titulos_regresion[vista]
        if vista in VISTAS_REGRESION
        else TITULOS_VISTA[vista]
    )
    ruta = (
        "Configuración"
        if vista == "configuracion"
        else f"Regresión / {titulo}"
        if vista in VISTAS_REGRESION
        else RUTAS_VISTA[vista]
    )
    nombre_dataset = escape(etiqueta)
    st.markdown(
        f"""
        <section class="atlas-header">
          <div class="atlas-breadcrumb">{APP_NAME} / {ruta}</div>
          <h1 class="atlas-title">{titulo}</h1>
          <p class="atlas-subtitle">Dataset activo: <strong>{nombre_dataset}</strong></p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if vista in {"datos", "eda"}:
        _resumen_calidad(datos)


def mostrar_faltantes(disponibilidad: Disponibilidad) -> None:
    """Presenta advertencias para cada prerrequisito faltante en la vista."""
    for mensaje in disponibilidad.faltantes:
        st.warning(mensaje)


def ir_a(destino: str) -> None:
    """Actualiza la vista activa en la sesión de Streamlit."""
    st.session_state["vista_activa"] = destino


def mostrar_guia(
    vista, familia=None, disponibilidad=None, configuracion=None
) -> None:
    """Muestra la orientación contextual de la etapa actual y acciones siguientes."""
    if familia is None:
        if vista == "configuracion":
            st.markdown("**Etapa 2 · Configuración**")
            return
        if vista == "datos":
            st.markdown("**Etapa 1 · Fuente activa**")
            st.caption(
                "Cargue los datos y configure un modelo. Limpieza global y partición manual son opcionales."
            )
            st.button(
                "Configurar tarea",
                key="flujo_configurar",
                on_click=ir_a,
                args=("configuracion",),
            )
        else:
            st.caption(
                "Exploración independiente: no requiere objetivo ni partición supervisada."
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
    st.markdown(f"**Etapa {etapa} de {nombre}**")
    if configuracion is not None:
        particion = (
            "compartida"
            if configuracion.get("particion_global") is not None
            else "automática al ejecutar"
        )
        with st.expander("Contexto activo"):
            st.caption(
                f"Objetivo: {configuracion['target']} · Predictores: {len(configuracion['features'])} · "
                f"Prueba: {configuracion['test_size']:.0%} · Semilla: {configuracion['random_state']} · Partición {particion}."
            )
    if configurando and familia == "clasificacion":
        st.caption(
            "Use clases discretas para clasificar y regresión para cantidades continuas. No se convierten objetivos automáticamente."
        )
    if disponibilidad is not None and not disponibilidad.lista:
        if configurando or consultando:
            mostrar_faltantes(disponibilidad)
    elif configurando:
        st.success(
            "Configuración utilizable. Continúe a un modelo o experimento."
        )
    elif not consultando:
        st.caption(
            "Ajuste los parámetros y ejecute. Editarlos no reemplaza la evidencia guardada."
        )
    destino = "configuracion"
    if consultando and not hay_evidencia(st.session_state, familia):
        st.info(
            "Falta ejecutar un modelo o experimento del contexto actual. Revise la configuración y continúe a un modelo."
        )
    if not configurando:
        st.button(
            "Revisar configuración",
            key="flujo_corregir",
            on_click=ir_a,
            args=(destino,),
        )
    elif disponibilidad is not None:
        destino = "regresion_ridge" if familia == "regresion" else "clasificacion"
        st.button(
            "Continuar a un modelo",
            key="flujo_continuar",
            disabled=not disponibilidad.lista,
            on_click=ir_a,
            args=(destino,),
        )
    if not consultando and hay_evidencia(st.session_state, familia):
        destino = (
            "lab3_resultados" if familia == "regresion" else "lab2_resultados"
        )
        st.button(
            "Revisar resultados guardados",
            key="flujo_resultados",
            on_click=ir_a,
            args=(destino,),
        )
