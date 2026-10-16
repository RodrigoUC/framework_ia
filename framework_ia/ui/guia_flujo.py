"""Orientación nativa: etapa actual, requisitos y siguiente acción concreta."""

import streamlit as st

from .flujo import Disponibilidad, hay_evidencia


def mostrar_faltantes(disponibilidad: Disponibilidad) -> None:
    for mensaje in disponibilidad.faltantes:
        st.warning(mensaje)


def ir_a(destino: str) -> None:
    st.session_state["vista_activa"] = destino


def mostrar_guia(vista, familia=None, disponibilidad=None, configuracion=None) -> None:
    st.caption(
        "Flujo de trabajo: 1. Fuente → 2. Configuración → 3. Ejecución → 4. Resultados"
    )
    if familia is None:
        if vista == "datos":
            st.markdown("**Etapa 1 · Fuente activa**")
            st.caption(
                "Revise columnas y calidad. La limpieza global y la partición manual son opcionales; no necesita aplicarlas para entrenar."
            )
            izquierda, derecha = st.columns(2)
            izquierda.button(
                "Configurar clasificación",
                key="flujo_config_clasif",
                on_click=ir_a,
                args=("clasificacion_configuracion",),
            )
            derecha.button(
                "Configurar regresión",
                key="flujo_config_reg",
                on_click=ir_a,
                args=("regresion_configuracion",),
            )
        else:
            st.caption(
                "Exploración independiente: EDA, ACP y clustering no requieren target ni partición supervisada. Configure las variables de esta vista antes de ejecutar."
            )
        return
    nombre = "regresión" if familia == "regresion" else "clasificación"
    configurando = vista in (
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
        st.caption(
            f"Objetivo: {configuracion['target']} · Predictores: {len(configuracion['features'])} · "
            f"Prueba: {configuracion['test_size']:.0%} · Semilla: {configuracion['random_state']} · Partición {particion}."
        )
    if configurando and familia == "clasificacion":
        st.caption(
            "Clasificación requiere etiquetas discretas. Si quiere predecir una cantidad continua, use Configuración de regresión; no se convierten automáticamente los objetivos."
        )
    if disponibilidad is not None and not disponibilidad.lista:
        st.caption(
            "Hay requisitos pendientes. No se ejecutarán modelos hasta resolverlos."
        )
        if configurando or consultando:
            mostrar_faltantes(disponibilidad)
    elif configurando:
        st.success(
            "Configuración utilizable. Puede abrir modelos o experimentos; cada ejecución verifica sus requisitos propios. No hace falta confirmar ni preparar globalmente los datos."
        )
    elif consultando:
        st.caption(
            "Está consultando evidencia guardada. Esta vista no vuelve a entrenar."
        )
    else:
        st.caption(
            "Revise los parámetros de esta vista y pulse su botón de ejecución. Cambiarlos no ejecuta ni reemplaza los resultados guardados."
        )
    destino = (
        f"{'regresion' if familia == 'regresion' else 'clasificacion'}_configuracion"
    )
    if consultando and not hay_evidencia(st.session_state, familia):
        st.info(
            "Falta ejecutar un modelo o experimento del contexto actual. Revise la configuración y continúe a un modelo."
        )
    if not configurando:
        st.button(
            f"Revisar configuración de {nombre}",
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
