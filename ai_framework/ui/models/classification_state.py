"""Supervised classification state and signatures independent from widgets."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from ai_framework.ui.models.session_state import (
    estado_para_familia,
    firma_parametros,
    huella_datos,
    state_value,
)


def classification_fingerprint(configuracion: dict, *parametros) -> tuple:
    """Build specific signature for supervised classification results."""
    particion = configuracion.get("particion_global")
    return (
        configuracion.get("tarea"),
        configuracion["target"],
        tuple(configuracion["features"]),
        configuracion["test_size"],
        configuracion["random_state"],
        configuracion["estratificar"],
        configuracion["incluir_categoricas"],
        configuracion["imputar"],
        configuracion["estandarizar"],
        particion.huella if particion is not None else None,
        *parametros,
    )


def current_classification_configuration(datos: pd.DataFrame, estado) -> dict | None:
    """Build effective classification configuration from saved controls."""
    estado = estado_para_familia(estado, "clasificacion")
    if len(datos.columns) < 2:
        return None
    objetivo = estado.get("clasif_target", datos.columns[-1])
    if objetivo not in datos.columns:
        objetivo = datos.columns[-1]
    features = [columna for columna in datos.columns if columna != objetivo]
    features_guardadas = estado.get("clasif_features", features)
    if any(columna not in features for columna in features_guardadas):
        features_guardadas = features
    particion = estado.get("particion_global")
    usar_particion = particion is not None and estado.get(
        "clasif_usar_particion_global", False
    )
    return {
        **(
            {"tarea": estado.get("sup_tarea")} if estado.get("sup_inicializada") else {}
        ),
        "target": objetivo,
        "features": features_guardadas,
        "test_size": particion.porcentaje_test
        if usar_particion
        else state_value(estado, "clasif_test_size", 25) / 100,
        "random_state": int(
            particion.semilla
            if usar_particion
            else state_value(estado, "clasif_random_state", 42)
        ),
        "estratificar": particion.columna_estratificacion is not None
        if usar_particion
        else state_value(estado, "clasif_strat", True),
        "incluir_categoricas": state_value(estado, "clasif_incluir_categoricas", False),
        "imputar": state_value(estado, "clasif_imputar", True),
        "estandarizar": state_value(estado, "clasif_escalar", True),
        "usar_particion_global": bool(usar_particion),
        "particion_global": particion if usar_particion else None,
    }


def sync_context(
    state: MutableMapping[str, Any], data: pd.DataFrame, config: dict
) -> tuple:
    """Discard previous results when data, target, features, or split change."""
    sig = (
        state.get("dataset_identidad"),
        huella_datos(data),
        *classification_fingerprint(config),
    )
    if state.get("clasificacion_contexto") != sig:
        for key in (
            "resultado_modelos_clasificacion",
            "preview_clasificacion",
            "resultado_lab2_experimento",
            "resultado_lab2_intento_experimento",
            "error_lab2_experimento",
            "resultado_lab2_modelo",
            "lab2_firma_experimento",
            "lab2_firma_modelo",
        ):
            state.pop(key, None)
        for key in tuple(state):
            if key.startswith(("resultado_lab2_modelo_", "contexto_lab2_modelo_")):
                state.pop(key, None)
        state["clasificacion_contexto"] = sig
    return sig


def clear_classification_results(state: MutableMapping[str, Any]) -> None:
    """Clear classification evidence when active data is not configurable."""
    for key in tuple(state):
        if key.startswith(
            (
                "resultado_modelos_clasificacion",
                "resultado_lab2_modelo",
                "contexto_lab2_modelo_",
                "lab2_firma_modelo",
                "resultado_lab2_experimento",
                "resultado_lab2_intento_experimento",
                "error_lab2_experimento",
                "lab2_firma_experimento",
            )
        ):
            state.pop(key, None)
    state.pop("clasificacion_contexto", None)


def sync_execution(
    state: MutableMapping[str, Any], kind: str, signature: str
) -> None:
    """Invalidate only the execution whose parameters have changed."""
    key = f"lab2_firma_{kind}"
    if state.get(key) != signature:
        state.pop(f"resultado_lab2_{kind}", None)
        state[key] = signature

# Backward compatibility aliases
sincronizar_contexto = sync_context
limpiar_resultados_clasificacion = clear_classification_results
sincronizar_ejecucion = sync_execution

# Backward compatibility aliases
firma_clasificacion = classification_fingerprint
configuracion_clasificacion_actual = current_classification_configuration
