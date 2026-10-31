"""Preprocesamiento reutilizable para algoritmos no supervisados y supervisados.

Este módulo aplica el patrón Strategy mediante configuraciones inmutables y
transformadores compatibles con scikit-learn. La selección, imputación,
codificación y escala se resuelven fuera de los modelos para mantener una sola
responsabilidad por clase.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..utils.results import DatosPreparados
from ..utils.sklearn_compatibility import crear_codificador_denso
from ..utils.data.preprocessing_config import ConfiguracionPreprocesamiento


class PreprocesadorNoSupervisado:
    """Construye matrices numéricas sin acoplarse a un algoritmo específico."""

    def __init__(self, configuracion: ConfiguracionPreprocesamiento) -> None:
        self._configuracion = configuracion

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
                        + [("codificar", crear_codificador_denso())]
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


class PreprocesadorSupervisado(TransformerMixin, BaseEstimator):
    """Transformador sklearn ajustable y clonable, con índices preservados.

    ``seleccion_k`` conserva los ``k`` atributos transformados con mayor
    información mutua respecto del objetivo. Requiere ``y`` en ``fit``.
    """

    def __init__(
        self,
        *,
        incluir_categoricas=False,
        imputar=True,
        estandarizar=True,
        seleccion_k=None,
        random_state=0,
    ):
        self.incluir_categoricas = incluir_categoricas
        self.imputar = imputar
        self.estandarizar = estandarizar
        self.seleccion_k = seleccion_k
        self.random_state = random_state

    @staticmethod
    def _tabla(datos):
        if not isinstance(datos, pd.DataFrame) or datos.empty:
            raise ValueError("Se requiere un DataFrame con al menos una fila.")
        if not datos.columns.is_unique:
            raise ValueError("Las columnas deben tener nombres únicos.")
        return datos.replace([np.inf, -np.inf], np.nan).copy()

    def _limpiar(self, datos):
        trabajo = self._tabla(datos)
        faltantes = [c for c in self.columnas_origen_ if c not in trabajo]
        if faltantes:
            raise ValueError(f"Faltan variables predictoras: {faltantes}")
        trabajo = trabajo.loc[:, list(self.columnas_origen_)]
        for columna in self.categoricas_:
            trabajo[columna] = (
                trabajo[columna]
                .map(lambda valor: str(valor) if pd.notna(valor) else np.nan)
                .astype(object)
            )
        if not self.imputar and trabajo.isna().any().any():
            raise ValueError(
                "Hay valores faltantes; active la imputación o limpie los datos."
            )
        return trabajo

    def fit(self, X, y=None):
        trabajo = self._tabla(X)
        self.feature_names_in_ = np.asarray(trabajo.columns, dtype=object)
        self.n_features_in_ = len(trabajo.columns)
        self.columnas_descartadas_ = tuple(
            c for c in trabajo if trabajo[c].nunique(dropna=True) <= 1
        )
        self.columnas_origen_ = tuple(
            c for c in trabajo if c not in self.columnas_descartadas_
        )
        if not self.columnas_origen_:
            raise ValueError(
                "Las variables de entrenamiento no contienen variación suficiente."
            )
        retenidas = trabajo.loc[:, list(self.columnas_origen_)]
        self.numericas_ = tuple(retenidas.select_dtypes(include="number").columns)
        self.categoricas_ = tuple(c for c in retenidas if c not in self.numericas_)
        if self.categoricas_ and not self.incluir_categoricas:
            raise TypeError(
                "Hay columnas categóricas seleccionadas. Active su codificación "
                f"o elimínelas: {list(self.categoricas_)}"
            )
        transformadores = []
        if self.numericas_:
            pasos = []
            if self.imputar:
                pasos.append(("imputar", SimpleImputer(strategy="median")))
            if self.estandarizar:
                pasos.append(("escalar", StandardScaler()))
            transformadores.append(
                (
                    "numericas",
                    Pipeline(pasos or [("passthrough", "passthrough")]),
                    list(self.numericas_),
                )
            )
        if self.categoricas_:
            pasos = []
            if self.imputar:
                pasos.append(("imputar", SimpleImputer(strategy="most_frequent")))
            pasos.append(("codificar", crear_codificador_denso()))
            transformadores.append(
                ("categoricas", Pipeline(pasos), list(self.categoricas_))
            )
        self.transformador_ = ColumnTransformer(transformadores, remainder="drop")
        self.transformador_.fit(self._limpiar(X), y)
        self.selector_ = None
        if self.seleccion_k is not None:
            if y is None:
                raise ValueError("La selección de atributos requiere el objetivo de train.")
            completas = np.asarray(
                self.transformador_.transform(self._limpiar(X)), dtype=float
            )
            k = int(self.seleccion_k)
            if k < 1:
                raise ValueError("seleccion_k debe ser un entero positivo.")
            y_arr = np.asarray(y)
            if pd.api.types.is_float_dtype(y_arr):
                scorer = partial(mutual_info_regression, random_state=self.random_state)
            else:
                scorer = partial(mutual_info_classif, random_state=self.random_state)
            self.selector_ = SelectKBest(
                scorer,
                k=min(k, completas.shape[1]),
            ).fit(completas, y_arr)
        return self

    def transform(self, X):
        check_is_fitted(self, "transformador_")
        trabajo = self._limpiar(X)
        valores = np.asarray(self.transformador_.transform(trabajo), dtype=float)
        if getattr(self, "selector_", None) is not None:
            valores = valores[:, self.selector_.get_support()]
        if not np.isfinite(valores).all():
            raise ValueError("El preprocesamiento produjo valores no finitos.")
        return pd.DataFrame(
            valores, index=X.index, columns=self.get_feature_names_out()
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformador_")
        nombres = self.transformador_.get_feature_names_out()
        if getattr(self, "selector_", None) is not None:
            return nombres[self.selector_.get_support()]
        return nombres

    def puntajes_informacion_mutua(self) -> pd.Series | None:
        """Información mutua de cada atributo transformado, calculada en train."""
        check_is_fitted(self, "transformador_")
        if getattr(self, "selector_", None) is None:
            return None
        return pd.Series(
            self.selector_.scores_,
            index=self.transformador_.get_feature_names_out(),
            name="informacion_mutua",
        ).sort_values(ascending=False)

    def ajustar_transformar(self, datos, y=None):
        matriz = self.fit(datos, y).transform(datos)
        return DatosPreparados(
            matriz=matriz,
            columnas_origen=self.columnas_origen_,
            columnas_descartadas=self.columnas_descartadas_,
            transformador=self,
        )


SupervisedPreprocessor = PreprocesadorSupervisado
UnsupervisedPreprocessor = PreprocesadorNoSupervisado
PreprocessingConfig = ConfiguracionPreprocesamiento

