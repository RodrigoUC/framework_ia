"""Configuración compartida de regresión, sin ajustar modelos al navegar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def configuracion_regresion_actual(datos: pd.DataFrame, estado) -> dict | None:
    """Lee los controles guardados con los mismos valores que la interfaz."""
    if len(datos.columns) < 2:
        return None
    target = estado.get("reg_target", datos.columns[-1])
    if target not in datos:
        target = datos.columns[-1]
    disponibles = [col for col in datos if col != target]
    numericas = datos[disponibles].select_dtypes(include="number").columns.tolist()
    features = estado.get("reg_features", numericas)
    if any(col not in disponibles for col in features):
        features = numericas
    simples = [col for col in features if col in numericas]
    simple = estado.get("reg_feature_simple")
    if simple not in simples:
        simple = simples[0] if simples else None
    particion = estado.get("particion_global")
    usar = particion is not None and estado.get("reg_usar_particion_global", False)
    return {
        "target": target,
        "features": list(features),
        "feature_simple": simple,
        "test_size": particion.porcentaje_test
        if usar
        else estado.get("reg_test_size", 25) / 100,
        "random_state": int(
            particion.semilla if usar else estado.get("reg_random_state", 42)
        ),
        "incluir_categoricas": estado.get("reg_incluir_categoricas", False),
        "imputar": estado.get("reg_imputar", True),
        "estandarizar": estado.get("reg_escalar", True),
        "usar_particion_global": bool(usar),
        "particion_global": particion if usar else None,
    }


def firma_regresion(configuracion: dict) -> tuple:
    """Incluye índices reales, además de la identidad declarada de la partición."""
    particion = configuracion.get("particion_global")
    indices = None
    if particion is not None:
        indices = tuple(
            tuple(getattr(particion, nombre).index)
            if getattr(particion, nombre) is not None
            else ()
            for nombre in ("train", "test", "validacion")
        )
    return (
        configuracion["target"],
        tuple(configuracion["features"]),
        configuracion["feature_simple"],
        configuracion["test_size"],
        configuracion["random_state"],
        configuracion["incluir_categoricas"],
        configuracion["imputar"],
        configuracion["estandarizar"],
        indices,
    )


def configurar_regresion(datos: pd.DataFrame) -> dict | None:
    """Expone target numérico, predictores, split y preparación train-only."""
    actual = configuracion_regresion_actual(datos, st.session_state)
    if actual is None:
        st.warning("Se necesita un target y al menos una variable predictora.")
        return None
    st.markdown("### Configuración de regresión")
    if st.session_state.get("reg_target") not in datos:
        st.session_state.pop("reg_target", None)
    target = st.selectbox(
        "Variable objetivo (target)",
        datos.columns.tolist(),
        index=len(datos.columns) - 1,
        key="reg_target",
        persist_state="session",
    )
    serie = datos[target]
    numerico = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(
        serie
    )
    valido = (
        numerico
        and not serie.isna().any()
        and np.isfinite(serie).all()
        and serie.nunique() > 1
    )
    st.caption(
        f"Target: {target} · Filas: {len(serie)} · Nulos: {int(serie.isna().sum())} · Valores distintos: {serie.nunique()}"
    )
    if not valido:
        st.warning(
            "El target debe ser numérico, finito, sin nulos y no constante. No se imputará ni se eliminarán sus filas automáticamente."
        )
    else:
        st.dataframe(serie.describe().rename("Valor").to_frame(), width="stretch")
    opciones = [col for col in datos if col != target]
    defecto = datos[opciones].select_dtypes(include="number").columns.tolist()
    if any(col not in opciones for col in st.session_state.get("reg_features", [])):
        st.session_state["reg_features"] = defecto
    features = st.multiselect(
        "Variables predictoras",
        opciones,
        default=defecto,
        key="reg_features",
        persist_state="session",
    )
    simples = [col for col in features if col in defecto]
    if st.session_state.get("reg_feature_simple") not in simples:
        st.session_state.pop("reg_feature_simple", None)
    st.selectbox(
        "Predictor de regresión lineal simple",
        simples,
        key="reg_feature_simple",
        persist_state="session",
        help="RLS utiliza solo este predictor; los demás modelos usan las variables seleccionadas.",
    )
    st.markdown("#### Partición y reproducibilidad")
    particion = st.session_state.get("particion_global")
    usar = False
    if particion is not None:
        usar = st.checkbox(
            "Usar partición calculada en Datos",
            key="reg_usar_particion_global",
            persist_state="session",
        )
    if usar:
        st.caption(
            f"Train: {len(particion.train)} · Test: {len(particion.test)} · Semilla: {particion.semilla}. Se conservan los índices de la partición aplicada."
        )
    else:
        st.slider(
            "Tamaño de prueba (%)",
            10,
            60,
            25,
            5,
            key="reg_test_size",
            persist_state="session",
        )
        st.number_input(
            "Random state",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
            key="reg_random_state",
            persist_state="session",
        )
    st.caption(
        "Regresión no estratifica por clases. El conjunto de prueba se reserva para la evaluación final."
    )
    st.markdown("#### Preprocesamiento aprendido solo con entrenamiento")
    st.checkbox(
        "Codificar categóricas",
        value=False,
        key="reg_incluir_categoricas",
        persist_state="session",
    )
    st.checkbox(
        "Imputar nulos de predictores",
        value=True,
        key="reg_imputar",
        persist_state="session",
    )
    st.checkbox(
        "Escalar predictores numéricos",
        value=True,
        key="reg_escalar",
        persist_state="session",
    )
    return configuracion_regresion_actual(datos, st.session_state)
