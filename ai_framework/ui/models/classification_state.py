"""Estado y firmas de clasificación supervisada independientes de los widgets."""

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


def firma_clasificacion(configuracion: dict, *parametros) -> tuple:
    """Construye la firma específica para resultados de clasificación supervisada."""
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


def configuracion_clasificacion_actual(datos: pd.DataFrame, estado) -> dict | None:
    """Construye la configuración efectiva de clasificación desde los controles guardados."""
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


def sincronizar_contexto(
    estado: MutableMapping[str, Any], datos: pd.DataFrame, configuracion: dict
) -> tuple:
    """Descarta resultados anteriores al cambiar datos, target, atributos o split."""
    firma = (
        estado.get("dataset_identidad"),
        huella_datos(datos),
        *firma_clasificacion(configuracion),
    )
    if estado.get("clasificacion_contexto") != firma:
        for clave in (
            "resultado_modelos_clasificacion",
            "preview_clasificacion",
            "resultado_lab2_experimento",
            "resultado_lab2_intento_experimento",
            "error_lab2_experimento",
            "resultado_lab2_modelo",
            "lab2_firma_experimento",
            "lab2_firma_modelo",
        ):
            estado.pop(clave, None)
        for clave in tuple(estado):
            if clave.startswith(("resultado_lab2_modelo_", "contexto_lab2_modelo_")):
                estado.pop(clave, None)
        estado["clasificacion_contexto"] = firma
    return firma


def limpiar_resultados_clasificacion(estado: MutableMapping[str, Any]) -> None:
    """Elimina la evidencia de clasificación cuando los datos activos no son configurables."""
    for clave in tuple(estado):
        if clave.startswith(
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
            estado.pop(clave, None)
    estado.pop("clasificacion_contexto", None)


def sincronizar_ejecucion(
    estado: MutableMapping[str, Any], tipo: str, firma: str
) -> None:
    """Invalida solo la ejecución cuyos parámetros han cambiado."""
    clave = f"lab2_firma_{tipo}"
    if estado.get(clave) != firma:
        estado.pop(f"resultado_lab2_{tipo}", None)
        estado[clave] = firma
