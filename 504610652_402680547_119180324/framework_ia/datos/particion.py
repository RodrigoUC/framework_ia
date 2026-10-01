"""División reproducible de un dataset completo en train/test/validación.

La partición vive en la capa de datos porque es una operación general sobre
filas, independiente de cualquier algoritmo. Las vistas supervisadas
(Clasificación, Regresión) pueden reutilizar el mismo resultado en lugar de
calcular su propia partición aleatoria.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..resultados import ResultadoParticion
from ..utils import ConfiguracionParticion


class Particionador:
    """Divide un DataFrame en train/test (y validación opcional) por filas."""

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
        porcentaje_resto = configuracion.porcentaje_test + configuracion.porcentaje_validacion
        if not 0 < porcentaje_resto < 1:
            raise ValueError("La suma de prueba y validación debe estar entre 0 y 1.")

        columna = configuracion.columna_estratificacion
        advertencias: list[str] = []

        estrato, advertencia = self._resolver_estrato(datos, columna)
        if advertencia:
            advertencias.append(advertencia)
        indice_train, indice_resto = train_test_split(
            datos.index,
            test_size=porcentaje_resto,
            random_state=configuracion.semilla,
            stratify=estrato,
        )

        indice_test = indice_resto
        validacion = None
        distribucion_validacion = None
        if configuracion.porcentaje_validacion > 0:
            proporcion_validacion = configuracion.porcentaje_validacion / porcentaje_resto
            estrato_resto, advertencia_resto = self._resolver_estrato(
                datos.loc[indice_resto], columna
            )
            if advertencia_resto:
                advertencias.append(advertencia_resto)
            indice_test, indice_validacion = train_test_split(
                indice_resto,
                test_size=proporcion_validacion,
                random_state=configuracion.semilla,
                stratify=estrato_resto,
            )
            validacion = datos.loc[indice_validacion].copy()
            distribucion_validacion = self._distribucion(validacion, columna)

        train = datos.loc[indice_train].copy()
        test = datos.loc[indice_test].copy()

        return ResultadoParticion(
            train=train,
            test=test,
            validacion=validacion,
            porcentaje_train=1 - porcentaje_resto,
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
