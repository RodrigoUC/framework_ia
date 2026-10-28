"""División reproducible de un dataset completo en train/test/validación.

La partición vive en la capa de datos porque es una operación general sobre
filas, independiente de cualquier algoritmo. Las vistas supervisadas
(Clasificación, Regresión) pueden reutilizar el mismo resultado en lugar de
calcular su propia partición aleatoria.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.results import ResultadoParticion
from ..utils.data.partition_config import ConfiguracionParticion


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


class Particionador:
    """Divide un DataFrame en train/test (y validación opcional) por filas."""

    huella_datos = staticmethod(huella_datos)
    validar_particion = staticmethod(validar_particion)
    dividir_indices = staticmethod(dividir_indices)
    comprobar_clases = staticmethod(comprobar_clases)

    def __init__(self, configuracion: ConfiguracionParticion) -> None:
        self._configuracion = configuracion

    def dividir(self, datos: pd.DataFrame) -> ResultadoParticion:
        """Calcula la partición y sus metadatos sobre el dataset recibido."""
        configuracion = self._configuracion
        if not isinstance(datos, pd.DataFrame) or datos.empty:
            raise ValueError("Se requiere un DataFrame con al menos una fila.")
        if not 0 < configuracion.porcentaje_test < 1:
            raise ValueError("El porcentaje de prueba debe estar entre 0 y 1.")
        if not 0 <= configuracion.porcentaje_validacion < 1:
            raise ValueError("El porcentaje de validación debe estar entre 0 y 1.")
        porcentaje_resto = round(
            configuracion.porcentaje_test + configuracion.porcentaje_validacion, 8
        )
        if not 0 < porcentaje_resto < 1:
            raise ValueError("La suma de prueba y validación debe estar entre 0 y 1.")

        columna = configuracion.columna_estratificacion
        advertencias: list[str] = []

        estrato, advertencia = self._resolver_estrato(datos, columna)
        if advertencia:
            advertencias.append(advertencia)
        posiciones_train, posiciones_resto = train_test_split(
            np.arange(len(datos)),
            test_size=porcentaje_resto,
            random_state=configuracion.semilla,
            stratify=estrato,
        )

        indice_train = datos.index.take(posiciones_train)
        indice_resto = datos.index.take(posiciones_resto)
        indice_test = indice_resto
        validacion = None
        distribucion_validacion = None
        if configuracion.porcentaje_validacion > 0:
            proporcion_validacion = round(
                configuracion.porcentaje_validacion / porcentaje_resto, 8
            )
            estrato_resto, advertencia_resto = self._resolver_estrato(
                datos.loc[indice_resto], columna
            )
            if advertencia_resto:
                advertencias.append(advertencia_resto)
            posiciones_test, posiciones_validacion = train_test_split(
                np.arange(len(indice_resto)),
                test_size=proporcion_validacion,
                random_state=configuracion.semilla,
                stratify=estrato_resto,
            )
            indice_test = indice_resto.take(posiciones_test)
            indice_validacion = indice_resto.take(posiciones_validacion)
            validacion = datos.loc[indice_validacion].copy()
            distribucion_validacion = self._distribucion(validacion, columna)

        train = datos.loc[indice_train].copy()
        test = datos.loc[indice_test].copy()

        return ResultadoParticion(
            train=train,
            test=test,
            validacion=validacion,
            porcentaje_train=round(1 - porcentaje_resto, 8),
            porcentaje_test=configuracion.porcentaje_test,
            porcentaje_validacion=configuracion.porcentaje_validacion,
            columna_estratificacion=columna,
            distribucion_train=self._distribucion(train, columna),
            distribucion_test=self._distribucion(test, columna),
            distribucion_validacion=distribucion_validacion,
            semilla=configuracion.semilla,
            advertencia=" ".join(dict.fromkeys(advertencias)) or None,
        )

    @staticmethod
    def _resolver_estrato(
        datos: pd.DataFrame, columna: str | None
    ) -> tuple[pd.Series | None, str | None]:
        """Valida si la columna indicada puede usarse para estratificar."""
        if columna is None:
            return None, None
        if columna not in datos.columns:
            raise KeyError(f"La columna de estratificación '{columna}' no existe.")
        serie = datos[columna]
        if serie.isna().any():
            return None, "Se ignoró la estratificación: la columna tiene valores nulos."
        conteos = serie.value_counts(dropna=False)
        if len(conteos) <= 1:
            return None, "Se ignoró la estratificación: la columna tiene una sola clase."
        if conteos.min() < 2:
            return None, (
                "Se ignoró la estratificación: alguna clase tiene una sola observación."
            )
        return serie, None

    @staticmethod
    def _distribucion(datos: pd.DataFrame, columna: str | None) -> dict | None:
        """Calcula la distribución de clases del subconjunto, si aplica."""
        if columna is None:
            return None
        return datos[columna].value_counts(dropna=False).to_dict()


PartitionConfig = ConfiguracionParticion
DataSplitter = Particionador

