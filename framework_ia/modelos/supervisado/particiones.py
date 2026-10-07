"""Validación estricta de particiones reutilizadas por modelos supervisados."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ...resultados import ResultadoParticion


def huella_datos(datos: pd.DataFrame) -> str:
    """Huella de valores, índices, columnas y tipos para reproducibilidad."""
    resumen = hashlib.sha256()
    resumen.update(pd.util.hash_pandas_object(datos, index=True).values.tobytes())
    resumen.update(repr(tuple(datos.columns)).encode())
    resumen.update(repr(tuple(map(str, datos.dtypes))).encode())
    return resumen.hexdigest()


def validar_particion(particion: ResultadoParticion, datos: pd.DataFrame) -> None:
    """Rechaza índices ambiguos, solapamiento, omisiones y snapshots obsoletos."""
    if not isinstance(particion, ResultadoParticion):
        raise TypeError("particion debe ser un ResultadoParticion.")
    if not datos.index.is_unique:
        raise ValueError(
            "El dataset necesita índices únicos para reutilizar una partición."
        )
    vistos = pd.Index([])
    for nombre, subconjunto in (
        ("train", particion.train),
        ("test", particion.test),
        ("validacion", particion.validacion),
    ):
        if subconjunto is None and nombre == "validacion":
            continue
        if not isinstance(subconjunto, pd.DataFrame) or subconjunto.empty:
            raise ValueError(f"La partición {nombre} está vacía.")
        if not subconjunto.index.is_unique:
            raise ValueError(f"La partición {nombre} tiene índices duplicados.")
        if not vistos.intersection(subconjunto.index).empty:
            raise ValueError("Las particiones tienen índices solapados.")
        if not subconjunto.index.isin(datos.index).all():
            raise ValueError("La partición contiene índices ajenos al dataset activo.")
        if not subconjunto.columns.equals(datos.columns):
            raise ValueError(
                "La partición está obsoleta: cambiaron las columnas del dataset."
            )
        actual = datos.loc[subconjunto.index]
        if not actual.equals(subconjunto):
            raise ValueError(
                "La partición está obsoleta: cambiaron los datos del dataset."
            )
        vistos = vistos.append(subconjunto.index)
    if len(vistos) != len(datos) or not datos.index.isin(vistos).all():
        raise ValueError("La partición debe cubrir todas las filas del dataset activo.")


def dividir_indices(base, target, *, test_size, random_state, stratify):
    """Divide sin tocar valores; errores de estratificación son explícitos."""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1 (excluido).")
    try:
        train, test = train_test_split(
            np.arange(len(base)),
            test_size=test_size,
            random_state=random_state,
            stratify=base[target] if stratify else None,
        )
    except ValueError as exc:
        raise ValueError(
            "No se pudo crear la partición. Revise el tamaño de los subconjuntos "
            f"y la cantidad de ejemplos por clase. Detalle: {exc}"
        ) from exc
    return base.index.take(train), base.index.take(test)


def comprobar_clases(y_train, *evaluaciones):
    if y_train.nunique() < 2:
        raise ValueError(
            "El conjunto de entrenamiento debe contener al menos dos clases."
        )
    for y in evaluaciones:
        if not y.isin(y_train.unique()).all():
            raise ValueError(
                "Hay clases en validación/prueba ausentes de entrenamiento; "
                "use una partición estratificada con más ejemplos por clase."
            )
