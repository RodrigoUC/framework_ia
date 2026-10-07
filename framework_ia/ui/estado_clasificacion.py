"""Estado de clasificación explícito y verificable, independiente de Streamlit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from .configuracion_clasificacion import firma_clasificacion


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


def sincronizar_contexto(
    estado: MutableMapping[str, Any], datos: pd.DataFrame, configuracion: dict
) -> tuple:
    """Descarta resultados anteriores al cambiar datos, target, atributos o split.

    No restaura resultados antiguos si el usuario vuelve a una configuración:
    debe ejecutar de nuevo para que las métricas siempre reflejen lo visible.
    """
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
            "resultado_lab2_modelo",
            "lab2_firma_experimento",
            "lab2_firma_modelo",
        ):
            estado.pop(clave, None)
        estado["clasificacion_contexto"] = firma
    return firma


def sincronizar_ejecucion(
    estado: MutableMapping[str, Any], tipo: str, firma: str
) -> None:
    """Invalida solo la ejecución cuyos parámetros han cambiado."""
    clave = f"lab2_firma_{tipo}"
    if estado.get(clave) != firma:
        estado.pop(f"resultado_lab2_{tipo}", None)
        estado[clave] = firma
