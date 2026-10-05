"""Preprocesamiento supervisado: se aprende exclusivamente de entrenamiento.

La limpieza, selección de columnas constantes, imputación, codificación y escala
se reutilizan sin volver a ajustar al transformar validación, prueba o inferencia.
El preprocesador no supervisado público conserva su comportamiento original.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ...resultados import DatosPreparados
from ...utils import crear_codificador_denso


class PreprocesadorSupervisado(TransformerMixin, BaseEstimator):
    """Transformador sklearn ajustable y clonable, con índices preservados."""

    def __init__(self, *, incluir_categoricas=False, imputar=True, estandarizar=True):
        self.incluir_categoricas = incluir_categoricas
        self.imputar = imputar
        self.estandarizar = estandarizar

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
        return self

    def transform(self, X):
        check_is_fitted(self, "transformador_")
        trabajo = self._limpiar(X)
        valores = np.asarray(self.transformador_.transform(trabajo), dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError("El preprocesamiento produjo valores no finitos.")
        return pd.DataFrame(
            valores, index=X.index, columns=self.get_feature_names_out()
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformador_")
        # Preserve prefixes to prevent collisions between original/dummy names.
        return self.transformador_.get_feature_names_out()

    def ajustar_transformar(self, datos):
        matriz = self.fit_transform(datos)
        return DatosPreparados(
            matriz=matriz,
            columnas_origen=self.columnas_origen_,
            columnas_descartadas=self.columnas_descartadas_,
            transformador=self,
        )
