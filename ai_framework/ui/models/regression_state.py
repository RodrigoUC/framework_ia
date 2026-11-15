"""Supervised regression state and signatures independent from widgets."""

from __future__ import annotations

import pandas as pd

from ai_framework.ui.models.session_state import (
    estado_para_familia,
    firma_parametros,
    huella_datos,
)


def current_regression_configuration(data: pd.DataFrame, state) -> dict | None:
    """Build effective regression configuration from saved controls."""
    state = estado_para_familia(state, "regresion")
    if len(data.columns) < 2:
        return None
    target = state.get("reg_target", data.columns[-1])
    if target not in data:
        target = data.columns[-1]
    available = [col for col in data if col != target]
    numeric = data[available].select_dtypes(include="number").columns.tolist()
    features = state.get("reg_features", numeric)
    if any(col not in available for col in features):
        features = numeric
    simples = [col for col in features if col in numeric]
    simple = state.get("reg_feature_simple")
    if simple not in simples:
        simple = simples[0] if simples else None
    partition = state.get("particion_global")
    use_partition = partition is not None and state.get("reg_usar_particion_global", False)
    return {
        **(
            {"tarea": state.get("sup_tarea")} if state.get("sup_inicializada") else {}
        ),
        "target": target,
        "features": list(features),
        "feature_simple": simple,
        "test_size": partition.porcentaje_test
        if use_partition
        else state.get("reg_test_size", 25) / 100,
        "random_state": int(
            partition.semilla if use_partition else state.get("reg_random_state", 42)
        ),
        "incluir_categoricas": state.get("reg_incluir_categoricas", False),
        "imputar": state.get("reg_imputar", True),
        "estandarizar": state.get("reg_escalar", True),
        "usar_particion_global": bool(use_partition),
        "particion_global": partition if use_partition else None,
    }


def regression_fingerprint(config: dict) -> tuple:
    """Build regression signature including actual partition indices."""
    partition = config.get("particion_global")
    indices = None
    if partition is not None:
        indices = tuple(
            tuple(getattr(partition, name).index)
            if getattr(partition, name) is not None
            else ()
            for name in ("train", "test", "validacion")
        )
    return (
        config.get("tarea"),
        config["target"],
        tuple(config["features"]),
        config["feature_simple"],
        config["test_size"],
        config["random_state"],
        config["incluir_categoricas"],
        config["imputar"],
        config["estandarizar"],
        indices,
    )


def clear_regression_results(state) -> None:
    """Clear snapshots, attempts, and errors without altering controls or splits."""
    for key in tuple(state):
        if key.startswith(
            ("resultado_lab3_", "contexto_lab3_", "error_lab3_", "lab3_firma_")
        ):
            state.pop(key, None)
    state.pop("regresion_contexto", None)


def sync_regression_context(state, data, config: dict) -> tuple:
    """Synchronize regression context and invalidate obsolete executions."""
    sig = (
        state.get("dataset_identidad"),
        huella_datos(data),
        firma_parametros(state.get("dataset_preparacion", {})),
        *regression_fingerprint(config),
    )
    if state.get("regresion_contexto") != sig:
        clear_regression_results(state)
        state["regresion_contexto"] = sig
    return sig

# Backward compatibility aliases
limpiar_resultados_regresion = clear_regression_results
sincronizar_contexto_regresion = sync_regression_context

# Backward compatibility aliases
configuracion_regresion_actual = current_regression_configuration
firma_regresion = regression_fingerprint
