"""Un único estado de preparación supervisada, independiente de cada estimador."""

import streamlit as st

from ..datos.eda import EDA

TAREAS = {"clasificacion": "Clasificación", "regresion": "Regresión"}
OPCIONES_COMUNES = (
    "target",
    "features",
    "test_size",
    "random_state",
    "incluir_categoricas",
    "imputar",
    "escalar",
    "usar_particion_global",
)


def inicializar_configuracion(datos, estado) -> None:
    """Migra una configuración explícita una sola vez; nunca deduce la tarea."""
    if not estado.get("sup_inicializada"):
        ruta = estado.get("vista_activa", "")
        tarea = estado.get("sup_tarea")
        if tarea is None and ruta in {"regresion", "regresion_configuracion"}:
            tarea = "regresion"
        elif tarea is None and ruta == "clasificacion_configuracion":
            tarea = "clasificacion"
        estado["sup_tarea"] = tarea
        prefijo = "reg" if tarea == "regresion" else "clasif"
        for opcion in (*OPCIONES_COMUNES, "feature_simple", "strat"):
            antigua = f"{prefijo}_{opcion}"
            if tarea is not None and antigua in estado:
                estado.setdefault(f"sup_{opcion}", estado[antigua])
        estado["sup_inicializada"] = True
    if len(datos.columns) < 2:
        return
    if estado.get("sup_target") not in datos:
        estado["sup_target"] = datos.columns[-1]
    disponibles = [col for col in datos if col != estado["sup_target"]]
    if "sup_features" not in estado or any(
        col not in disponibles for col in estado["sup_features"]
    ):
        estado["sup_features"] = disponibles
    simples = datos[estado["sup_features"]].select_dtypes(include="number").columns
    if estado.get("sup_feature_simple") not in simples:
        estado["sup_feature_simple"] = simples[0] if len(simples) else None
    for opcion, valor in {
        "test_size": 25,
        "random_state": 42,
        "incluir_categoricas": False,
        "imputar": True,
        "escalar": True,
        "usar_particion_global": False,
        "strat": True,
    }.items():
        estado.setdefault(f"sup_{opcion}", valor)
    if estado.get("particion_global") is not None:
        firma = firma_preparacion(estado)
        anterior = estado.get("sup_particion_firma")
        if anterior is not None and anterior != firma:
            estado.pop("particion_global", None)
            estado.pop("sup_particion_firma", None)
            estado["sup_usar_particion_global"] = False
            estado["sup_aviso_particion"] = (
                "La configuración cambió. Se descartó la partición anterior; use división automática o calcule otra."
            )
        else:
            estado["sup_particion_firma"] = firma


def firma_preparacion(estado) -> tuple:
    return tuple(
        tuple(estado.get(f"sup_{clave}", []))
        if clave == "features"
        else estado.get(f"sup_{clave}")
        for clave in (
            "tarea",
            "target",
            "features",
            "test_size",
            "random_state",
            "incluir_categoricas",
            "imputar",
            "escalar",
            "strat",
            "feature_simple",
        )
    )


def estado_para_familia(estado, familia):
    """Proyecta el estado canónico sobre contratos anteriores sin copiar widgets."""
    if not estado.get("sup_inicializada") and not any(
        f"sup_{opcion}" in estado for opcion in OPCIONES_COMUNES
    ):
        return estado
    prefijo = "reg" if familia == "regresion" else "clasif"
    proyectado = dict(estado)
    for opcion in (*OPCIONES_COMUNES, "feature_simple", "strat"):
        clave = f"sup_{opcion}"
        if clave in estado:
            proyectado[f"{prefijo}_{opcion}"] = estado[clave]
    return proyectado


def _usar_particion() -> None:
    particion = st.session_state.get("particion_global")
    if particion is not None and st.session_state.get("sup_usar_particion_global"):
        st.session_state["sup_test_size"] = round(particion.porcentaje_test * 100)
        st.session_state["sup_random_state"] = particion.semilla
        st.session_state["sup_strat"] = particion.columna_estratificacion is not None
        st.session_state["sup_particion_firma"] = firma_preparacion(st.session_state)


def configurar_supervisado(datos) -> None:
    """Prepara datos y protocolo; los botones de entrenar pertenecen a cada modelo."""
    tarea = st.radio(
        "Tarea",
        list(TAREAS),
        index=None,
        format_func=TAREAS.get,
        key="sup_tarea",
        persist_state="session",
        horizontal=True,
    )
    if len(datos.columns) < 2:
        st.warning("Se necesita un objetivo y al menos una variable predictora.")
        return
    if tarea is None:
        st.info("Seleccione Clasificación para clases o Regresión para cantidades.")
        return
    target = st.selectbox(
        "Variable objetivo (target)",
        datos.columns.tolist(),
        key="sup_target",
        persist_state="session",
    )
    st.multiselect(
        "Variables predictoras",
        [col for col in datos if col != target],
        key="sup_features",
        persist_state="session",
    )
    with st.expander("Distribución del objetivo"):
        serie = datos[target]
        st.caption(f"Filas: {len(serie)} · Nulos: {int(serie.isna().sum())}")
        if tarea == "clasificacion":
            resumen, _ = EDA(dataframe=datos).balance_objetivo(target)
            if resumen.empty:
                resumen = serie.value_counts(dropna=False).rename("Cantidad").to_frame()
        else:
            resumen = serie.describe().rename("Valor").to_frame()
        st.dataframe(resumen, width="stretch")
    if tarea == "regresion":
        simples = datos[st.session_state["sup_features"]].select_dtypes(
            include="number"
        )
        st.selectbox(
            "Predictor de regresión lineal simple",
            simples.columns.tolist(),
            key="sup_feature_simple",
            persist_state="session",
        )
    st.markdown("#### Partición y reproducibilidad")
    if st.session_state.get("sup_aviso_particion"):
        st.info(st.session_state["sup_aviso_particion"])
    externa = bool(
        st.session_state.get("particion_global") is not None
        and st.session_state.get("sup_usar_particion_global")
    )
    st.slider(
        "Tamaño de prueba (%)",
        10,
        60,
        step=5,
        key="sup_test_size",
        persist_state="session",
        disabled=externa,
    )
    st.number_input(
        "Semilla",
        min_value=0,
        max_value=10_000,
        step=1,
        key="sup_random_state",
        persist_state="session",
        disabled=externa,
    )
    if tarea == "clasificacion":
        st.checkbox(
            "Estratificar por target",
            key="sup_strat",
            persist_state="session",
            disabled=externa,
        )
    if st.session_state.get("particion_global") is not None:
        st.checkbox(
            "Usar partición calculada",
            key="sup_usar_particion_global",
            persist_state="session",
            on_change=_usar_particion,
        )
    st.markdown("#### Preparación aprendida solo con entrenamiento")
    for etiqueta, clave in (
        ("Codificar categóricas", "incluir_categoricas"),
        ("Imputar nulos de predictores", "imputar"),
        ("Escalar predictores numéricos", "escalar"),
    ):
        st.checkbox(etiqueta, key=f"sup_{clave}", persist_state="session")
