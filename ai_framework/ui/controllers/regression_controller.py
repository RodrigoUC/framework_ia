"""Controller for instantiation and execution snapshots for regression."""

from __future__ import annotations

from copy import deepcopy

import streamlit as st

from ai_framework.models.supervised import Regresion
from ai_framework.ui.models.session_state import firma_parametros, huella_datos


def argumentos_comunes(configuracion: dict) -> dict:
    """Build common partition and preprocessing arguments for regression."""
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


def create_regressor(datos, configuracion, *, algoritmo=None) -> Regresion:
    """Instantiate regression domain model with corresponding predictors."""
    features = (
        [configuracion["feature_simple"]]
        if algoritmo == "RLS"
        else configuracion["features"]
    )
    return Regresion(
        dataframe=datos, target=configuracion["target"], features=features
    )


def snapshot_ejecucion(resultado, datos, configuracion, parametros) -> dict:
    """Freeze context and provenance of a regression execution for export."""
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

crear_regresor = create_regressor
