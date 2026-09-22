"""Preprocesamiento reutilizable para algoritmos no supervisados.

Este módulo aplica el patrón Strategy mediante una configuración inmutable.
La selección, imputación, codificación y escala se resuelven fuera de los
modelos para mantener una sola responsabilidad por clase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from resultados import DatosPreparados


@dataclass(frozen=True)
class ConfiguracionPreprocesamiento:
    """Opciones para convertir un dataset arbitrario en una matriz numérica."""

    columnas: tuple[str, ...] | None = None
    incluir_categoricas: bool = False
    imputar: bool = True
    estandarizar: bool = True


class PreprocesadorNoSupervisado:
    """Construye matrices numéricas sin acoplarse a un algoritmo específico."""

    def __init__(self, configuracion: ConfiguracionPreprocesamiento) -> None:
        self._configuracion = configuracion

    @staticmethod
    def _crear_codificador() -> OneHotEncoder:
        """Crea un codificador denso compatible con distintas versiones."""
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def ajustar_transformar(self, datos: pd.DataFrame) -> DatosPreparados:
        """Valida, limpia y transforma el dataset seleccionado."""
        if not isinstance(datos, pd.DataFrame) or datos.empty:
            raise ValueError("Se requiere un DataFrame con al menos una fila.")

        seleccionadas = self._seleccionar_columnas(datos)
        trabajo = datos.loc[:, seleccionadas].copy()
        trabajo = trabajo.replace([np.inf, -np.inf], np.nan)

        descartadas = tuple(
            columna
            for columna in trabajo.columns
            if trabajo[columna].nunique(dropna=True) <= 1
        )
        if descartadas:
            trabajo = trabajo.drop(columns=list(descartadas))
        if trabajo.empty:
            raise ValueError(
                "Las columnas seleccionadas no contienen variación suficiente."
            )

        numericas = trabajo.select_dtypes(include="number").columns.tolist()
        categoricas = trabajo.select_dtypes(exclude="number").columns.tolist()
        if categoricas and not self._configuracion.incluir_categoricas:
            raise TypeError(
                "Hay columnas categóricas seleccionadas. Active su codificación "
                f"o elimínelas de la selección: {categoricas}"
            )

        transformadores = []
        if numericas:
            pasos_numericos: list[tuple[str, object]] = []
            if self._configuracion.imputar:
                pasos_numericos.append(("imputar", SimpleImputer(strategy="median")))
            if self._configuracion.estandarizar:
                pasos_numericos.append(("escalar", StandardScaler()))
            if not pasos_numericos:
                pasos_numericos.append(("passthrough", "passthrough"))
            transformadores.append(
                ("numericas", Pipeline(pasos_numericos), numericas)
            )

        if categoricas:
            pasos_categoricos: list[tuple[str, object]] = []
            if self._configuracion.imputar:
                pasos_categoricos.append(
                    ("imputar", SimpleImputer(strategy="most_frequent"))
                )
            transformadores.append(
                (
                    "categoricas",
                    Pipeline(
                        pasos_categoricos
                        + [("codificar", self._crear_codificador())]
                    ),
                    categoricas,
                )
            )

        transformador = ColumnTransformer(
            transformers=transformadores,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        matriz_np = transformador.fit_transform(trabajo)
        nombres = [
            nombre.split("__", maxsplit=1)[-1]
            for nombre in transformador.get_feature_names_out()
        ]
        matriz = pd.DataFrame(
            np.asarray(matriz_np, dtype=float),
            index=trabajo.index,
            columns=nombres,
        )
        if not np.isfinite(matriz.to_numpy()).all():
            raise ValueError("El preprocesamiento produjo valores no finitos.")

        return DatosPreparados(
            matriz=matriz,
            columnas_origen=tuple(trabajo.columns),
            columnas_descartadas=descartadas,
            transformador=transformador,
        )

    def _seleccionar_columnas(self, datos: pd.DataFrame) -> list[str]:
        """Resuelve las columnas configuradas y valida su existencia."""
        configuradas = self._configuracion.columnas
        if configuradas is None:
            if self._configuracion.incluir_categoricas:
                seleccionadas = datos.columns.tolist()
            else:
                seleccionadas = datos.select_dtypes(include="number").columns.tolist()
        else:
            seleccionadas = list(dict.fromkeys(configuradas))

        inexistentes = [
            columna for columna in seleccionadas if columna not in datos.columns
        ]
        if inexistentes:
            raise KeyError(f"Columnas inexistentes: {inexistentes}")
        if not seleccionadas:
            raise ValueError("Seleccione al menos una columna para el análisis.")
        return seleccionadas
