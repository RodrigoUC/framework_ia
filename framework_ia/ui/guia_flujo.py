"""Orientación nativa: etapa actual, requisitos y siguiente acción concreta."""

import streamlit as st

from .flujo import Disponibilidad, hay_evidencia


def mostrar_faltantes(disponibilidad: Disponibilidad) -> None:
    for mensaje in disponibilidad.faltantes:
        st.warning(mensaje)


def ir_a(destino: str) -> None:
    st.session_state["vista_activa"] = destino


def mostrar_guia(vista, familia=None, disponibilidad=None, configuracion=None) -> None:
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
        destino = "lab3_resultados" if familia == "regresion" else "lab2_resultados"
        st.button(
            "Revisar resultados guardados",
            key="flujo_resultados",
            on_click=ir_a,
            args=(destino,),
        )
