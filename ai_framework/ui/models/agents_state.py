"""Invalidation of agent sessions, independent of the widgets."""

from __future__ import annotations

from ai_framework.ui.models.session_state import firma_parametros, huella_datos

EDA_KEYS = ("agent_eda_session", "agent_eda_error", "agent_eda_context")
CLUSTERING_KEYS = (
    "agent_clustering_session",
    "agent_clustering_error",
    "agent_clustering_context",
)
SUPERVISED_KEYS = {
    family: (f"agent_{family}_session", f"agent_{family}_error", f"agent_{family}_context")
    for family in ("regression", "classification")
}


def _clear(state, keys) -> None:
    for key in keys:
        state.pop(key, None)


def clear_agent_results(state) -> None:
    """Discards blackboards, findings and conversations without touching the controls."""
    _clear(state, (*EDA_KEYS, *CLUSTERING_KEYS, *(k for keys in SUPERVISED_KEYS.values() for k in keys)))


def sync_eda_agent_context(state, data) -> tuple:
    """An EDA agent session is only valid for the dataset it was started with."""
    signature = (state.get("dataset_identidad"), huella_datos(data))
    if state.get("agent_eda_context") != signature:
        _clear(state, EDA_KEYS)
        state["agent_eda_context"] = signature
    return signature


def sync_clustering_agent_context(state, data, config: dict) -> tuple:
    """Changing variables, scaling or the sample invalidates the recommended partition."""
    signature = (
        state.get("dataset_identidad"),
        huella_datos(data),
        tuple(config["features"]),
        config["incluir_categoricas"],
        config["estandarizar"],
    )
    if state.get("agent_clustering_context") != signature:
        _clear(state, CLUSTERING_KEYS)
        state["agent_clustering_context"] = signature
    return signature


def _config_signature(config: dict) -> tuple:
    """Every control plus the indices of an external partition, if one is used."""
    partition = config.get("particion_global")
    indices = None
    if partition is not None:
        indices = tuple(
            tuple(getattr(partition, name).index) if getattr(partition, name) is not None else ()
            for name in ("train", "test", "validacion")
        )
    return firma_parametros({k: v for k, v in config.items() if k != "particion_global"}), indices


def sync_supervised_agent_context(state, data, config: dict, family: str) -> tuple:
    """Changing the target, features or split invalidates the recommended model."""
    session_key, _, context_key = SUPERVISED_KEYS[family]
    signature = (state.get("dataset_identidad"), huella_datos(data), _config_signature(config))
    if state.get(context_key) != signature:
        _clear(state, SUPERVISED_KEYS[family])
        state[context_key] = signature
    return signature
