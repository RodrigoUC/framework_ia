"""Configuración compartida de regresión, sin ajustar modelos al navegar."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .configuracion_supervisada import (
    configurar_supervisado,
    estado_para_familia,
    inicializar_configuracion,
)


def configuracion_regresion_actual(datos: pd.DataFrame, estado) -> dict | None:
    """Lee los controles guardados con los mismos valores que la interfaz."""
    estado = estado_para_familia(estado, "regresion")
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
        **(
            {"tarea": estado.get("sup_tarea")} if estado.get("sup_inicializada") else {}
        ),
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
        configuracion.get("tarea"),
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
    """Compatibilidad: muestra la configuracion unica, sin entrenar."""
    inicializar_configuracion(datos, st.session_state)
    configurar_supervisado(datos)
    return configuracion_regresion_actual(datos, st.session_state)
