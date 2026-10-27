"""Gestión de estado de sesión, huellas de datos y firmas de ejecución."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from framework_ia.ui.modelos.catalogo_vistas import OPCIONES_COMUNES


def huella_datos(datos: pd.DataFrame) -> str:
    """Detecta cambios de valores, orden, índice, nombres y tipos de columna."""
    firma = hashlib.sha256(
        pd.util.hash_pandas_object(datos, index=True).values.tobytes()
    )
    firma.update(repr(tuple(zip(datos.columns, datos.dtypes.astype(str)))).encode())
    return firma.hexdigest()


def firma_parametros(parametros: dict) -> str:
    """Normaliza configuraciones anidadas para comparar resultados vigentes."""
    return json.dumps(parametros, sort_keys=True, ensure_ascii=False, default=str)


def firma_preparacion(estado) -> tuple:
    """Construye la firma de configuración supervisada compartida."""
    return tuple(
        tuple(estado.get(f"sup_{clave}", []))
        if clave == "features"
        else estado.get(f"sup_{clave}")
        for clave in (
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


def inicializar_configuracion(datos: pd.DataFrame, estado) -> None:
    """Inicializa la configuración supervisada compartida sin deducir la tarea."""
    if not estado.get("sup_inicializada"):
        ruta = estado.get("vista_activa", "")
        tarea = estado.get("sup_tarea")
        if tarea is None and ruta in {"regresion", "regresion_configuracion"}:
            tarea = "regresion"
        elif tarea is None and ruta == "clasificacion_configuracion":
            tarea = "clasificacion"
        estado["sup_tarea"] = tarea
        prefijo = "reg" if tarea == "regresion" else "clasif"
        for opcion in (*OPCIONES_COMUNES, "feature_simple", "strat"):
            antigua = f"{prefijo}_{opcion}"
            if tarea is not None and antigua in estado:
                estado.setdefault(f"sup_{opcion}", estado[antigua])
        estado["sup_inicializada"] = True
    if len(datos.columns) < 2:
        return
    if estado.get("sup_target") not in datos:
        estado["sup_target"] = datos.columns[-1]
    disponibles = [col for col in datos if col != estado["sup_target"]]
    if "sup_features" not in estado or any(
        col not in disponibles for col in estado["sup_features"]
    ):
        estado["sup_features"] = disponibles
    simples = datos[estado["sup_features"]].select_dtypes(include="number").columns
    if estado.get("sup_feature_simple") not in simples:
        estado["sup_feature_simple"] = simples[0] if len(simples) else None
    for opcion, valor in {
        "test_size": 25,
        "random_state": 42,
        "incluir_categoricas": False,
        "imputar": True,
        "escalar": True,
        "usar_particion_global": False,
        "strat": True,
    }.items():
        estado.setdefault(f"sup_{opcion}", valor)
    if estado.get("particion_global") is not None:
        firma = firma_preparacion(estado)
        anterior = estado.get("sup_particion_firma")
        if anterior is not None and anterior != firma:
            estado.pop("particion_global", None)
            estado.pop("sup_particion_firma", None)
            estado["sup_usar_particion_global"] = False
            estado["sup_aviso_particion"] = (
                "La configuración cambió. Se descartó la partición anterior; use división automática o calcule otra."
            )
        else:
            estado["sup_particion_firma"] = firma


def estado_para_familia(estado, familia: str):
    """Proyecta el estado canónico sobre las claves específicas de cada familia."""
    if not estado.get("sup_inicializada") and not any(
        f"sup_{opcion}" in estado for opcion in OPCIONES_COMUNES
    ):
        return estado
    prefijo = "reg" if familia == "regresion" else "clasif"
    proyectado = dict(estado)
    for opcion in (*OPCIONES_COMUNES, "feature_simple", "strat"):
        clave = f"sup_{opcion}"
        if clave in estado:
            proyectado[f"{prefijo}_{opcion}"] = estado[clave]
    return proyectado


def state_value(estado, clave: str, predeterminado):
    """Lee un valor del estado de sesión con respaldo en el valor predeterminado."""
    return estado.get(clave, predeterminado)


def _firma(configuracion: dict, *parametros) -> tuple:
    """Identifica cuándo un resultado no supervisado sigue correspondiendo a los controles."""
    return (
        tuple(configuracion["features"]),
        configuracion["incluir_categoricas"],
        configuracion["estandarizar"],
        configuracion["filas"],
        *parametros,
    )


def _guardar_resultado(clave: str, firma: tuple, resultado) -> None:
    """Persiste un resultado entre los reruns de Streamlit."""
    st.session_state[clave] = {"firma": firma, "resultado": resultado}


def _recuperar_resultado(clave: str, firma: tuple):
    """Retorna el resultado únicamente si coincide con la configuración actual."""
    contenedor = st.session_state.get(clave)
    if contenedor and contenedor["firma"] == firma:
        return contenedor["resultado"]
    return None
