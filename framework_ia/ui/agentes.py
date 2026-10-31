"""Vistas de los agentes de cada sección; trabajan sobre el dataset activo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..agentes import (
    CLUSTERING_SOLICITADO,
    DATASET_CARGADO,
    AgenteClustering,
    AgenteEDA,
    BusEventos,
    Evento,
    Pizarra,
)
from ..agentes.llm import crear_proveedor
from ..resultados import SugerenciaModelado
from ..visualizacion import VisualizadorNoSupervisado
from .configuracion_agentes import configurar_agente_clustering, configurar_agente_eda
from .estado_agentes import sincronizar_contexto_agentes, sincronizar_contexto_clustering

ICONOS = {"critico": ":material/error:", "aviso": ":material/warning:", "info": ":material/info:"}
ORDEN_NIVELES = {"critico": 0, "aviso": 1, "info": 2}


def _iniciar(datos: pd.DataFrame, etiqueta: str, opciones: dict) -> None:
    llm = crear_proveedor(opciones["proveedor"], opciones["modelo"], opciones["api_key"])
    # El origen llega al LLM: sólo el nombre del archivo, nunca la ruta local.
    pizarra = Pizarra(datos, target=opciones["target"], origen=etiqueta)
    bus = BusEventos()
    agente = AgenteEDA(pizarra, llm).conectar(bus)
    with st.spinner("El agente está analizando el dataset…"):
        bus.emitir(Evento(DATASET_CARGADO))
    st.session_state["agentes_sesion"] = {
        "opciones": {k: v for k, v in opciones.items() if k != "api_key"},
        "pizarra": pizarra,
        "agente": agente,
        "bus": bus,
    }


def _aplicar_sugerencia(sugerencia: SugerenciaModelado) -> None:
    st.session_state["sup_tarea"] = sugerencia.tarea
    st.session_state["sup_target"] = sugerencia.target
    st.session_state["sup_features"] = list(sugerencia.features)
    st.session_state["vista_activa"] = "configuracion"


def _render_hallazgos(pizarra: Pizarra) -> None:
    conteo = pd.Series([h.nivel for h in pizarra.hallazgos], dtype=object).value_counts()
    a, b, c = st.columns(3)
    a.metric("Críticos", int(conteo.get("critico", 0)))
    b.metric("Avisos", int(conteo.get("aviso", 0)))
    c.metric("Informativos", int(conteo.get("info", 0)))
    for hallazgo in sorted(pizarra.hallazgos, key=lambda h: ORDEN_NIVELES[h.nivel]):
        mostrar = {"critico": st.error, "aviso": st.warning, "info": st.info}[hallazgo.nivel]
        mostrar(hallazgo.mensaje, icon=ICONOS[hallazgo.nivel])
    if not pizarra.hallazgos:
        st.success("El agente no encontró riesgos con sus reglas.", icon=":material/check_circle:")


def _render_siguiente_paso(agente: AgenteEDA) -> None:
    sugerencia = agente.sugerir_configuracion()
    st.subheader("Siguiente paso propuesto")
    if sugerencia is None:
        st.caption("Elija una variable objetivo para que el agente proponga el modelado.")
        return
    tarea = "regresión" if sugerencia.tarea == "regresion" else "clasificación"
    st.markdown(
        f"Tarea detectada: **{tarea}** sobre `{sugerencia.target}` "
        f"con {len(sugerencia.features)} predictoras."
    )
    if sugerencia.excluidas:
        st.caption("Excluye: " + ", ".join(f"`{c}`" for c in sugerencia.excluidas))
    st.button(
        "Aplicar configuración sugerida", key="agentes_aplicar", icon=":material/arrow_forward:",
        on_click=_aplicar_sugerencia, args=(sugerencia,),
        help="Abre Configuración con tarea, target y predictoras ya elegidas; no entrena.",
    )


def _render_conversacion(
    sesion: dict, *, clave: str = "agentes_pregunta", error: str = "error_agentes"
) -> None:
    pizarra, agente = sesion["pizarra"], sesion["agente"]
    st.subheader("Conversación")
    if agente.llm is None:
        st.caption("Configure un modelo de lenguaje para que el agente interprete y responda preguntas.")
        return
    for autor, texto in pizarra.mensajes:
        with st.chat_message("user" if autor == "usuario" else "assistant"):
            st.markdown(texto)
    pregunta = st.chat_input("Pregunte al agente sobre el dataset", key=clave)
    if pregunta:
        try:
            with st.spinner("Pensando…"):
                agente.preguntar(pregunta)
        except Exception as fallo:  # pylint: disable=broad-except
            st.session_state[error] = f"El modelo de lenguaje no respondió: {fallo}"
        st.rerun()


def render_agente_eda(datos: pd.DataFrame, etiqueta: str) -> None:
    st.caption(
        "El agente actúa por su cuenta al iniciarse (reglas deterministas) y luego "
        "responde preguntas con el modelo de lenguaje. Al LLM sólo se envía un "
        "resumen del dataset, nunca sus filas."
    )
    sincronizar_contexto_agentes(st.session_state, datos)
    opciones = configurar_agente_eda(datos)
    if opciones is not None:
        st.session_state.pop("error_agentes", None)
        try:
            _iniciar(datos, etiqueta, opciones)
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["error_agentes"] = f"No fue posible iniciar el agente: {error}"
    if error := st.session_state.get("error_agentes"):
        st.error(error)
    sesion = st.session_state.get("agentes_sesion")
    if sesion is None:
        st.info("Pulse «Iniciar agente» para analizar el dataset.", icon=":material/smart_toy:")
        return
    st.divider()
    st.subheader("Hallazgos")
    _render_hallazgos(sesion["pizarra"])
    _render_siguiente_paso(sesion["agente"])
    _render_conversacion(sesion)


def _iniciar_clustering(datos: pd.DataFrame, configuracion: dict, etiqueta: str, opciones: dict) -> None:
    llm = crear_proveedor(opciones["proveedor"], opciones["modelo"], opciones["api_key"])
    pizarra = Pizarra(datos, origen=etiqueta)
    bus = BusEventos()
    agente = AgenteClustering(
        pizarra,
        llm,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    ).conectar(bus)
    with st.spinner("El agente está comparando agrupamientos…"):
        bus.emitir(Evento(CLUSTERING_SOLICITADO))
    st.session_state["agente_clustering_sesion"] = {"pizarra": pizarra, "agente": agente, "bus": bus}


def _render_particion_recomendada(sesion: dict, mostrar_figura) -> None:
    pizarra = sesion["pizarra"]
    sugerencia = sesion["agente"].sugerir_agrupamiento()
    resultado = pizarra.resultados["resultado_cluster"]
    st.subheader("Partición recomendada")
    a, b, c = st.columns(3)
    a.metric("Algoritmo", sugerencia.algoritmo)
    b.metric("Clusters", sugerencia.n_clusters)
    c.metric("Silhouette", f"{sugerencia.silhouette:.4f}")
    mostrar_figura(VisualizadorNoSupervisado.clusters(resultado))
    mostrar_figura(VisualizadorNoSupervisado.perfiles_cluster(resultado))
    salida = pizarra.datos.loc[resultado.etiquetas.index].copy()
    salida["cluster"] = resultado.etiquetas
    st.download_button(
        "Descargar asignaciones",
        salida.to_csv(index=False).encode("utf-8"),
        file_name="clusters_agente.csv",
        mime="text/csv",
        key="agente_clustering_descarga",
    )
    with st.expander("Comparación evaluada"):
        kmeans = pizarra.resultados["evaluacion_kmeans"]
        ward = pizarra.resultados["evaluacion_ward"]
        st.dataframe(kmeans, width="stretch")
        mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(kmeans, "K-Means"))
        if not ward.empty:
            st.dataframe(ward, width="stretch")
            mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(ward, "HAC-ward"))


def render_agente_clustering(
    datos: pd.DataFrame, configuracion: dict, etiqueta: str, mostrar_figura
) -> None:
    """Agrupa sólo al pulsar el botón, con las variables de la sección Clustering."""
    if len(datos) < 3 or not configuracion["features"]:
        st.warning("Se requieren variables y al menos tres filas.")
        return
    sincronizar_contexto_clustering(st.session_state, datos, configuracion)
    opciones = configurar_agente_clustering()
    if opciones is not None:
        st.session_state.pop("error_agente_clustering", None)
        try:
            _iniciar_clustering(datos, configuracion, etiqueta, opciones)
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["error_agente_clustering"] = f"No fue posible iniciar el agente: {error}"
    if error := st.session_state.get("error_agente_clustering"):
        st.error(error)
    sesion = st.session_state.get("agente_clustering_sesion")
    if sesion is None:
        st.info("Pulse «Iniciar agente» para comparar agrupamientos.", icon=":material/smart_toy:")
        return
    st.divider()
    st.subheader("Hallazgos")
    _render_hallazgos(sesion["pizarra"])
    _render_particion_recomendada(sesion, mostrar_figura)
    _render_conversacion(sesion, clave="agente_clustering_pregunta", error="error_agente_clustering")
