"""Controlador de instanciación y snapshots de ejecución para regresión."""

from __future__ import annotations

from copy import deepcopy

import streamlit as st

from framework_ia.modelos.supervisado import Regresion
from framework_ia.ui.modelos.estado_sesion import firma_parametros, huella_datos


def argumentos_comunes(configuracion: dict) -> dict:
    """Construye los argumentos comunes de partición y preprocesamiento para regresión."""
    return {
        **{
            clave: configuracion[clave]
            for clave in (
                "test_size",
                "random_state",
                "incluir_categoricas",
                "imputar",
                "estandarizar",
            )
        },
        "particion": configuracion.get("particion_global"),
    }


def crear_regresor(datos, configuracion, *, algoritmo=None) -> Regresion:
    """Instancia el modelo de dominio de regresión con los predictores correspondientes."""
    features = (
        [configuracion["feature_simple"]]
        if algoritmo == "RLS"
        else configuracion["features"]
    )
    return Regresion(
        dataframe=datos, target=configuracion["target"], features=features
    )


def snapshot_ejecucion(resultado, datos, configuracion, parametros) -> dict:
    """Congela contexto y procedencia de una ejecución de regresión para exportación."""
    argumentos = {
        clave: deepcopy(valor)
        for clave, valor in configuracion.items()
        if clave != "particion_global"
    }
    argumentos["parametros_solicitados"] = deepcopy(parametros)
    fuente = deepcopy(st.session_state.get("fuente_metadata", {}))
    fuente["preparacion_previa"] = deepcopy(
        st.session_state.get("dataset_preparacion", {})
    )
    fuente["huella_datos_efectivos"] = huella_datos(datos)
    fuente.setdefault(
        "etiqueta",
        st.session_state.get("fuente_aplicada", (None, "Dataset activo"))[1],
    )
    return {
        "resultado": resultado,
        "argumentos": argumentos,
        "dataset": fuente,
        "firma_parametros": firma_parametros(parametros),
    }
