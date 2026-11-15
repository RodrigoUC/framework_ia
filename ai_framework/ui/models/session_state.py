"""Session state management, data fingerprints, and execution signatures."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from ai_framework.ui.models.view_catalog import OPCIONES_COMUNES


def data_fingerprint(data: pd.DataFrame) -> str:
    """Detect changes in values, ordering, index, names, and column types."""
    sig = hashlib.sha256(
        pd.util.hash_pandas_object(data, index=True).values.tobytes()
    )
    sig.update(repr(tuple(zip(data.columns, data.dtypes.astype(str)))).encode())
    return sig.hexdigest()


def parameters_fingerprint(parameters: dict) -> str:
    """Normalize nested configurations to compare valid results."""
    return json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)


def preparation_fingerprint(state) -> tuple:
    """Build signature for shared supervised configuration."""
    return tuple(
        tuple(state.get(f"sup_{key}", []))
        if key == "features"
        else state.get(f"sup_{key}")
        for key in (
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


def initialize_configuration(data: pd.DataFrame, state) -> None:
    """Initialize shared supervised configuration without inferring task."""
    if not state.get("sup_inicializada"):
        route = state.get("vista_activa", "")
        task = state.get("sup_tarea")
        if task is None and route in {"regresion", "regresion_configuracion"}:
            task = "regresion"
        elif task is None and route == "clasificacion_configuracion":
            task = "clasificacion"
        state["sup_tarea"] = task
        prefix = "reg" if task == "regresion" else "clasif"
        for opt in (*OPCIONES_COMUNES, "feature_simple", "strat"):
            old = f"{prefix}_{opt}"
            if task is not None and old in state:
                state.setdefault(f"sup_{opt}", state[old])
        state["sup_inicializada"] = True
    if len(data.columns) < 2:
        return
    if state.get("sup_target") not in data:
        state["sup_target"] = data.columns[-1]
    available = [col for col in data if col != state["sup_target"]]
    if "sup_features" not in state or any(
        col not in available for col in state["sup_features"]
    ):
        state["sup_features"] = available
    simples = data[state["sup_features"]].select_dtypes(include="number").columns
    if state.get("sup_feature_simple") not in simples:
        state["sup_feature_simple"] = simples[0] if len(simples) else None
    for opt, val in {
        "test_size": 25,
        "random_state": 42,
        "incluir_categoricas": False,
        "imputar": True,
        "escalar": True,
        "usar_particion_global": False,
        "strat": True,
    }.items():
        state.setdefault(f"sup_{opt}", val)
    if state.get("particion_global") is not None:
        sig = preparation_fingerprint(state)
        previous = state.get("sup_particion_firma")
        if previous is not None and previous != sig:
            state.pop("particion_global", None)
            state.pop("sup_particion_firma", None)
            state["sup_usar_particion_global"] = False
            state["sup_aviso_particion"] = (
                "La configuración cambió. Se descartó la partición anterior; use división automática o calcule otra."
            )
        else:
            state["sup_particion_firma"] = sig


def state_for_family(state, family: str):
    """Project canonical state onto family-specific keys."""
    if not state.get("sup_inicializada") and not any(
        f"sup_{opt}" in state for opt in OPCIONES_COMUNES
    ):
        return state
    prefix = "reg" if family == "regresion" else "clasif"
    projected = dict(state)
    for opt in (*OPCIONES_COMUNES, "feature_simple", "strat"):
        key = f"sup_{opt}"
        if key in state:
            projected[f"{prefix}_{opt}"] = state[key]
    return projected


def state_value(state, key: str, default):
    """Read value from session state with fallback to default."""
    return state.get(key, default)


def _fingerprint(config: dict, *parameters) -> tuple:
    """Identify whether unsupervised result still corresponds to controls."""
    return (
        tuple(config["features"]),
        config["incluir_categoricas"],
        config["estandarizar"],
        config["filas"],
        *parameters,
    )


def _save_result(key: str, signature: tuple, result) -> None:
    """Persist result across Streamlit reruns."""
    st.session_state[key] = {"firma": signature, "resultado": result}


def _retrieve_result(key: str, signature: tuple):
    """Return result only if it matches current configuration."""
    container = st.session_state.get(key)
    if container and container["firma"] == signature:
        return container["resultado"]
    return None

# Backward compatibility aliases
huella_datos = data_fingerprint
inicializar_configuracion = initialize_configuration

# Backward compatibility aliases
firma_parametros = parameters_fingerprint
firma_preparacion = preparation_fingerprint
estado_para_familia = state_for_family
_firma = _fingerprint
_guardar_resultado = _save_result
_recuperar_resultado = _retrieve_result
