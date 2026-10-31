"""Vistas de los agentes de IA de cada sección; trabajan sobre el dataset activo."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.agents import Blackboard, EDAAgent
from ai_framework.ui.controllers.agents_controller import (
    aplicar_sugerencia_modelado,
    iniciar_agente_clustering,
    iniciar_agente_eda,
    proveedor_por_defecto,
)
from ai_framework.ui.controllers.unsupervised_controller import _muestrear
from ai_framework.ui.models.agents_state import (
    sincronizar_contexto_agentes,
    sincronizar_contexto_clustering,
)
from ai_framework.ui.views.theme import _mostrar_figura
from ai_framework.utils.agents import (
    NOMBRES_PROVEEDORES,
    PROVEEDOR_SIN_LLM,
    PROVEEDORES_LLM,
)
from ai_framework.utils.visualization import VisualizadorNoSupervisado

SIN_OBJETIVO = "__sin_objetivo__"
ICONOS = {"critico": ":material/error:", "aviso": ":material/warning:", "info": ":material/info:"}
ORDEN_NIVELES = {"critico": 0, "aviso": 1, "info": 2}


def _nombre_proveedor(codigo: str) -> str:
    if codigo == PROVEEDOR_SIN_LLM:
        return t("agente_sin_llm", NOMBRES_PROVEEDORES[codigo])
    return NOMBRES_PROVEEDORES[codigo]


def _controles_llm() -> dict:
    """Proveedor, modelo y API key; deben dibujarse dentro de un formulario."""
    izquierda, derecha = st.columns(2)
    proveedor = izquierda.selectbox(
        t("agente_llm", "Modelo de lenguaje"),
        list(NOMBRES_PROVEEDORES),
        format_func=_nombre_proveedor,
        key="agentes_proveedor",
    )
    configuracion = PROVEEDORES_LLM.get(proveedor)
    modelo = derecha.text_input(
        t("agente_modelo", "Modelo"),
        value=os.environ.get("AGENTES_MODELO", ""),
        placeholder=(configuracion.modelo_sugerido if configuracion else "")
        or t("agente_modelo_placeholder", "nombre del modelo"),
        key="agentes_modelo",
        help=t("agente_modelo_help", "Vacío: AGENTES_MODELO o la sugerencia del proveedor."),
    )
    api_key = None
    if configuracion and configuracion.variable_clave:
        variable = configuracion.variable_clave
        api_key = st.text_input(
            t("agente_api_key", "API key"),
            type="password",
            key="agentes_api_key",
            placeholder=t("agente_key_detectada", "{variable} detectada", variable=variable)
            if os.environ.get(variable)
            else t("agente_key_definir", "o defina {variable}", variable=variable),
            help=t(
                "agente_api_key_help",
                "Sólo vive en esta sesión. Preferible: .streamlit/secrets.toml.",
            ),
        )
    return {"proveedor": proveedor, "modelo": modelo, "api_key": api_key}


def _boton_iniciar() -> bool:
    return st.form_submit_button(
        t("agente_iniciar", "Iniciar agente"), type="primary", icon=":material/smart_toy:"
    )


def _configurar_agente_eda(datos: pd.DataFrame) -> dict | None:
    """Devuelve la configuración sólo cuando se pulsa «Iniciar agente»."""
    st.session_state.setdefault("agentes_proveedor", proveedor_por_defecto())
    with st.form("form_agentes"):
        st.caption(
            t(
                "agente_eda_descripcion",
                "Perfila el dataset, detecta riesgos y sugiere la configuración de modelado.",
            )
        )
        columnas = [SIN_OBJETIVO, *map(str, datos.columns)]
        actual = str(st.session_state.get("sup_target", datos.columns[-1]))
        objetivo = st.selectbox(
            t("agente_objetivo", "Variable objetivo"),
            columnas,
            index=columnas.index(actual) if actual in columnas else len(columnas) - 1,
            format_func=lambda columna: t("agente_sin_objetivo", "(sin variable objetivo)")
            if columna == SIN_OBJETIVO
            else columna,
            key="agentes_target",
        )
        opciones = _controles_llm()
        iniciar = _boton_iniciar()
    if not iniciar:
        return None
    return {"target": None if objetivo == SIN_OBJETIVO else objetivo, **opciones}


def _configurar_agente_clustering() -> dict | None:
    """Usa las variables de la sección Clustering; sólo el botón agrupa."""
    st.session_state.setdefault("agentes_proveedor", proveedor_por_defecto())
    with st.form("form_agente_clustering"):
        st.caption(
            t(
                "agente_clustering_descripcion",
                "Compara K-Means y HAC, recomienda el número de clusters y describe cada grupo.",
            )
        )
        opciones = _controles_llm()
        iniciar = _boton_iniciar()
    return opciones if iniciar else None


def _render_hallazgos(pizarra: Blackboard) -> None:
    st.subheader(t("agente_hallazgos", "Hallazgos"))
    conteo = pd.Series([h.nivel for h in pizarra.hallazgos], dtype=object).value_counts()
    a, b, c = st.columns(3)
    a.metric(t("agente_criticos", "Críticos"), int(conteo.get("critico", 0)))
    b.metric(t("agente_avisos", "Avisos"), int(conteo.get("aviso", 0)))
    c.metric(t("agente_informativos", "Informativos"), int(conteo.get("info", 0)))
    for hallazgo in sorted(pizarra.hallazgos, key=lambda h: ORDEN_NIVELES[h.nivel]):
        mostrar = {"critico": st.error, "aviso": st.warning, "info": st.info}[hallazgo.nivel]
        mostrar(hallazgo.mensaje, icon=ICONOS[hallazgo.nivel])
    if not pizarra.hallazgos:
        st.success(
            t("agente_sin_riesgos", "El agente no encontró riesgos con sus reglas."),
            icon=":material/check_circle:",
        )


def _render_siguiente_paso(agente: EDAAgent) -> None:
    sugerencia = agente.sugerir_configuracion()
    st.subheader(t("agente_siguiente_paso", "Siguiente paso propuesto"))
    if sugerencia is None:
        st.caption(
            t(
                "agente_elija_objetivo",
                "Elija una variable objetivo para que el agente proponga el modelado.",
            )
        )
        return
    tarea = (
        t("nav_regresion", "Regresión")
        if sugerencia.tarea == "regresion"
        else t("nav_clasificacion", "Clasificación")
    )
    st.markdown(
        t(
            "agente_tarea_detectada",
            "Tarea detectada: **{tarea}** sobre `{target}` con {cantidad} predictoras.",
            tarea=tarea,
            target=sugerencia.target,
            cantidad=len(sugerencia.features),
        )
    )
    if sugerencia.excluidas:
        st.caption(
            t(
                "agente_excluye",
                "Excluye: {columnas}",
                columnas=", ".join(f"`{c}`" for c in sugerencia.excluidas),
            )
        )
    st.button(
        t("agente_aplicar", "Aplicar configuración sugerida"),
        key="agentes_aplicar",
        icon=":material/arrow_forward:",
        on_click=aplicar_sugerencia_modelado,
        args=(st.session_state, sugerencia),
        help=t(
            "agente_aplicar_help",
            "Abre Configuración con tarea, target y predictoras ya elegidas; no entrena.",
        ),
    )


def _render_conversacion(sesion: dict, *, clave: str, error: str) -> None:
    pizarra, agente = sesion["pizarra"], sesion["agente"]
    st.subheader(t("agente_conversacion", "Conversación"))
    if agente.llm is None:
        st.caption(
            t(
                "agente_configure_llm",
                "Configure un modelo de lenguaje para que el agente interprete y responda preguntas.",
            )
        )
        return
    for autor, texto in pizarra.mensajes:
        with st.chat_message("user" if autor == "usuario" else "assistant"):
            st.markdown(texto)
    pregunta = st.chat_input(
        t("agente_pregunta", "Pregunte al agente sobre el dataset"), key=clave
    )
    if pregunta:
        try:
            with st.spinner(t("agente_pensando", "Pensando…")):
                agente.preguntar(pregunta)
        except Exception as fallo:  # pylint: disable=broad-except
            st.session_state[error] = t(
                "agente_llm_error",
                "El modelo de lenguaje no respondió: {detalle}",
                detalle=fallo,
            )
        st.rerun()


def _render_particion_recomendada(sesion: dict) -> None:
    pizarra = sesion["pizarra"]
    sugerencia = sesion["agente"].sugerir_agrupamiento()
    resultado = pizarra.resultados["resultado_cluster"]
    st.subheader(t("agente_particion", "Partición recomendada"))
    a, b, c = st.columns(3)
    a.metric(t("agente_algoritmo", "Algoritmo"), sugerencia.algoritmo)
    b.metric(t("agente_clusters", "Clusters"), sugerencia.n_clusters)
    c.metric("Silhouette", f"{sugerencia.silhouette:.4f}")
    _mostrar_figura(VisualizadorNoSupervisado.clusters(resultado))
    _mostrar_figura(VisualizadorNoSupervisado.perfiles_cluster(resultado))
    salida = pizarra.datos.loc[resultado.etiquetas.index].copy()
    salida["cluster"] = resultado.etiquetas
    st.download_button(
        t("agente_descargar", "Descargar asignaciones"),
        salida.to_csv(index=False).encode("utf-8"),
        file_name="clusters_agente.csv",
        mime="text/csv",
        key="agente_clustering_descarga",
    )
    with st.expander(t("agente_comparacion", "Comparación evaluada")):
        kmeans = pizarra.resultados["evaluacion_kmeans"]
        ward = pizarra.resultados["evaluacion_ward"]
        st.dataframe(kmeans, width="stretch")
        _mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(kmeans, "K-Means"))
        if not ward.empty:
            st.dataframe(ward, width="stretch")
            _mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(ward, "HAC-ward"))


def _mostrar_error(clave: str) -> None:
    if error := st.session_state.get(clave):
        st.error(error)


def render_agente_eda(datos: pd.DataFrame, etiqueta: str) -> None:
    """Analiza sólo al pulsar el botón; navegar a la vista no consume cuota del LLM."""
    st.caption(
        t(
            "agente_eda_intro",
            "El agente actúa por su cuenta al iniciarse (reglas deterministas) y luego "
            "responde preguntas con el modelo de lenguaje. Al LLM sólo se envía un "
            "resumen del dataset, nunca sus filas.",
        )
    )
    sincronizar_contexto_agentes(st.session_state, datos)
    opciones = _configurar_agente_eda(datos)
    if opciones is not None:
        st.session_state.pop("error_agentes", None)
        try:
            with st.spinner(t("agente_analizando", "El agente está analizando el dataset…")):
                st.session_state["agentes_sesion"] = iniciar_agente_eda(datos, etiqueta, opciones)
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["error_agentes"] = t(
                "agente_error_inicio", "No fue posible iniciar el agente: {detalle}", detalle=error
            )
    _mostrar_error("error_agentes")
    sesion = st.session_state.get("agentes_sesion")
    if sesion is None:
        st.info(
            t("agente_eda_pendiente", "Pulse «Iniciar agente» para analizar el dataset."),
            icon=":material/smart_toy:",
        )
        return
    st.divider()
    _render_hallazgos(sesion["pizarra"])
    _render_siguiente_paso(sesion["agente"])
    _render_conversacion(sesion, clave="agentes_pregunta", error="error_agentes")


def render_agente_clustering(datos: pd.DataFrame, configuracion: dict, etiqueta: str) -> None:
    """Agrupa sólo al pulsar el botón, con las variables de la sección Clustering."""
    if len(datos) < 3 or not configuracion["features"]:
        st.warning(t("warn_requiere_variables", "Se requieren variables y al menos tres filas."))
        return
    muestra = _muestrear(datos, configuracion["filas"])
    sincronizar_contexto_clustering(st.session_state, muestra, configuracion)
    opciones = _configurar_agente_clustering()
    if opciones is not None:
        st.session_state.pop("error_agente_clustering", None)
        try:
            with st.spinner(t("agente_comparando", "El agente está comparando agrupamientos…")):
                st.session_state["agente_clustering_sesion"] = iniciar_agente_clustering(
                    muestra, configuracion, etiqueta, opciones
                )
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["error_agente_clustering"] = t(
                "agente_error_inicio", "No fue posible iniciar el agente: {detalle}", detalle=error
            )
    _mostrar_error("error_agente_clustering")
    sesion = st.session_state.get("agente_clustering_sesion")
    if sesion is None:
        st.info(
            t("agente_clustering_pendiente", "Pulse «Iniciar agente» para comparar agrupamientos."),
            icon=":material/smart_toy:",
        )
        return
    st.divider()
    _render_hallazgos(sesion["pizarra"])
    _render_particion_recomendada(sesion)
    _render_conversacion(
        sesion, clave="agente_clustering_pregunta", error="error_agente_clustering"
    )
