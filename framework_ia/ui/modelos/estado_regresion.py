"""Estado y firmas de regresión supervisada independientes de los widgets."""

from __future__ import annotations

import pandas as pd

from framework_ia.ui.modelos.estado_sesion import (
    estado_para_familia,
    firma_parametros,
    huella_datos,
)


def configuracion_regresion_actual(datos: pd.DataFrame, estado) -> dict | None:
    """Construye la configuración efectiva de regresión desde los controles guardados."""
    estado = estado_para_familia(estado, "regresion")
    if len(datos.columns) < 2:
        return None
    target = estado.get("reg_target", datos.columns[-1])
    if target not in datos:
        target = datos.columns[-1]
    disponibles = [col for col in datos if col != target]
    numericas = datos[disponibles].select_dtypes(include="number").columns.tolist()
    features = estado.get("reg_features", numericas)
    if any(col not in disponibles for col in features):
        features = numericas
    simples = [col for col in features if col in numericas]
    simple = estado.get("reg_feature_simple")
    if simple not in simples:
        simple = simples[0] if simples else None
    particion = estado.get("particion_global")
    usar = particion is not None and estado.get("reg_usar_particion_global", False)
    return {
        **(
            {"tarea": estado.get("sup_tarea")} if estado.get("sup_inicializada") else {}
        ),
        "target": target,
        "features": list(features),
        "feature_simple": simple,
        "test_size": particion.porcentaje_test
        if usar
        else estado.get("reg_test_size", 25) / 100,
        "random_state": int(
            particion.semilla if usar else estado.get("reg_random_state", 42)
        ),
        "incluir_categoricas": estado.get("reg_incluir_categoricas", False),
        "imputar": estado.get("reg_imputar", True),
        "estandarizar": estado.get("reg_escalar", True),
        "usar_particion_global": bool(usar),
        "particion_global": particion if usar else None,
    }


def firma_regresion(configuracion: dict) -> tuple:
    """Construye la firma de regresión incluyendo los índices reales de partición."""
    particion = configuracion.get("particion_global")
    indices = None
    if particion is not None:
        indices = tuple(
            tuple(getattr(particion, nombre).index)
            if getattr(particion, nombre) is not None
            else ()
            for nombre in ("train", "test", "validacion")
        )
    return (
        configuracion.get("tarea"),
        configuracion["target"],
        tuple(configuracion["features"]),
        configuracion["feature_simple"],
        configuracion["test_size"],
        configuracion["random_state"],
        configuracion["incluir_categoricas"],
        configuracion["imputar"],
        configuracion["estandarizar"],
        indices,
    )


def limpiar_resultados_regresion(estado) -> None:
    """Limpia snapshots, intentos y errores sin alterar controles ni particiones."""
    for clave in tuple(estado):
        if clave.startswith(
            ("resultado_lab3_", "contexto_lab3_", "error_lab3_", "lab3_firma_")
        ):
            estado.pop(clave, None)
    estado.pop("regresion_contexto", None)


def sincronizar_contexto_regresion(estado, datos, configuracion: dict) -> tuple:
    """Sincroniza el contexto de regresión e invalida ejecuciones obsoletas."""
    firma = (
        estado.get("dataset_identidad"),
        huella_datos(datos),
        firma_parametros(estado.get("dataset_preparacion", {})),
        *firma_regresion(configuracion),
    )
    if estado.get("regresion_contexto") != firma:
        limpiar_resultados_regresion(estado)
        estado["regresion_contexto"] = firma
    return firma
