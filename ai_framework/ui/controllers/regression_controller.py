"""Controller for instantiation and execution snapshots for regression."""

from __future__ import annotations

from copy import deepcopy

import streamlit as st

from ai_framework.models.supervised import Regresion
from ai_framework.ui.models.session_state import firma_parametros, huella_datos


def common_arguments(config: dict) -> dict:
    """Build common partition and preprocessing arguments for regression."""
    return {
        **{
            key: config[key]
            for key in (
                "test_size",
                "random_state",
                "incluir_categoricas",
                "imputar",
                "estandarizar",
            )
        },
        "particion": config.get("particion_global"),
    }


def create_regressor(data, config, *, algorithm=None, **kwargs) -> Regresion:
    """Instantiate regression domain model with corresponding predictors."""
    alg = algorithm or kwargs.get("algoritmo")
    features = (
        [config["feature_simple"]]
        if alg == "RLS"
        else config["features"]
    )
    return Regresion(
        dataframe=data, target=config["target"], features=features
    )


def execution_snapshot(result, data, config, parameters) -> dict:
    """Freeze context and provenance of a regression execution for export."""
    arguments = {
        key: deepcopy(value)
        for key, value in config.items()
        if key != "particion_global"
    }
    arguments["parametros_solicitados"] = deepcopy(parameters)
    source = deepcopy(st.session_state.get("fuente_metadata", {}))
    source["preparacion_previa"] = deepcopy(
        st.session_state.get("dataset_preparacion", {})
    )
    source["huella_datos_efectivos"] = huella_datos(data)
    source.setdefault(
        "etiqueta",
        st.session_state.get("fuente_aplicada", (None, "Active dataset"))[1],
    )
    return {
        "resultado": result,
        "argumentos": arguments,
        "dataset": source,
        "firma_parametros": firma_parametros(parameters),
    }

crear_regresor = create_regressor

# Backward compatibility aliases
argumentos_comunes = common_arguments
snapshot_ejecucion = execution_snapshot
crear_regresor = create_regressor
