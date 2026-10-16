"""Controles y firmas compartidos por clasificación individual y LAB02.

Los controles conservan sus valores al navegar. Este módulo solo coordina la
interfaz; el particionado y el ajuste de transformadores pertenecen al dominio.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from .configuracion_supervisada import (
    configurar_supervisado,
    estado_para_familia,
    inicializar_configuracion,
)


def configurar_clasificacion(
    datos: pd.DataFrame, mostrar_figura: Callable | None = None
) -> dict | None:
    """Compatibilidad: muestra la configuracion unica, sin entrenar."""
    inicializar_configuracion(datos, st.session_state)
    configurar_supervisado(datos)
    return configuracion_clasificacion_actual(datos, st.session_state)


def firma_clasificacion(configuracion: dict, *parametros) -> tuple:
    """Firma específica para resultados supervisados."""
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
    """Build the effective configuration from saved controls without rendering widgets."""
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


def state_value(estado, clave: str, predeterminado):
    """Read a widget value with the same default used by its renderer."""
    return estado.get(clave, predeterminado)
