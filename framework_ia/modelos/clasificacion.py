"""Clasificación supervisada con separación de métricas y resultados."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from ..datos.preprocesamiento import ConfiguracionPreprocesamiento, PreprocesadorNoSupervisado
from ..resultados import ResultadoClasificacion
from .supervisado import Supervisado


class Clasificacion(Supervisado):
    """Clasificación supervisada con Random Forest y Naive Bayes."""

    def __init__(
        self,
        dataframe=None,
        ruta_datos=None,
        *,
        target=None,
        features=None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataframe=dataframe,
            ruta_datos=ruta_datos,
            target=target,
            features=features,
            **kwargs,
        )
        self.resultado: ResultadoClasificacion | None = None
        self.modelo = None
        self._preparados = None
        self.exactitud = None

    def entrenar(
        self,
        *,
        algoritmo: str,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = True,
        incluir_categoricas: bool = False,
        imputar: bool = True,
        estandarizar: bool = True,
        **kwargs,
    ) -> ResultadoClasificacion:
        """Prepara datos, separa train/test y entrena el clasificador."""
        self._definir_target()
        self._definir_features()
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1 (excluido).")
        if self._tipo_target() != "clasificacion":
            raise TypeError("La variable objetivo debe ser categórica para clasificación.")
        if self.datos.empty:
            raise ValueError("No hay datos para entrenar.")

        objetivo = self.datos[self.target]
        columnas = list(self.features)
        if self.target in columnas:
            raise ValueError("El target no debe incluirse entre las features.")
        if not columnas:
            raise ValueError("Debe seleccionar al menos una variable predictora.")

        base = pd.concat([self.datos[columnas], objetivo], axis=1).copy()
        base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.target])
        if base.empty:
            raise ValueError("No hay filas válidas para entrenar tras depurar nulos.")

        X = base[self.features]
        y = base[self.target].reset_index(drop=True)
        X = X.reset_index(drop=True)

        preprocesador = PreprocesadorNoSupervisado(
            ConfiguracionPreprocesamiento(
                columnas=tuple(X.columns),
                incluir_categoricas=incluir_categoricas,
                imputar=imputar,
                estandarizar=estandarizar,
            )
        )
        preparados = preprocesador.ajustar_transformar(X)
        self._preparados = preparados

        estrato = (
            y
            if stratify and y.nunique(dropna=False) > 1
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            preparados.matriz,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=estrato,
        )
        if X_train.empty or X_test.empty:
            raise ValueError("La partición quedó vacía; ajuste el porcentaje de prueba.")

        if algoritmo == "RF":
            modelo = RandomForestClassifier(random_state=random_state, **kwargs)
        elif algoritmo == "NR":
            if "random_state" in kwargs:
                kwargs.pop("random_state")
            modelo = GaussianNB(**kwargs)
        else:
            raise ValueError("Algoritmo no soportado: use RF o NR.")

        modelo.fit(X_train, y_train)
        y_pred = pd.Series(modelo.predict(X_test), name="prediccion", index=y_test.index)
        self.modelo = modelo
        self.exactitud = float(accuracy_score(y_test, y_pred))

        labels = self._ordenar_labels(y_test, y_pred)
        matriz_confusion = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=pd.Index(labels, name="real"),
            columns=pd.Index(labels, name="prediccion"),
        )

        metricas = self._metricas_supervisadas(
            y_train=y_train,
            y_true=y_test,
            y_pred=y_pred,
            matriz=matriz_confusion,
            labels=labels,
        )
        self.resultado = ResultadoClasificacion(
            algoritmo=algoritmo,
            target=self.target,
            features=tuple(self.features),
            labels=tuple(labels),
            y_true=pd.Series(y_test, name="real", copy=True),
            y_pred=y_pred.copy(),
            matriz_confusion=matriz_confusion,
            metricas=metricas,
            muestra_train=len(X_train),
            muestra_test=len(X_test),
            modelo=modelo,
        )
        return self.resultado

    def predecir(self, X=None):
        """Predice la clase para un conjunto o el test de la última ejecución."""
        if self.modelo is None:
            raise RuntimeError("No hay un modelo entrenado.")
        if X is None:
            if self.resultado is None:
                raise RuntimeError("No hay predicciones disponibles.")
            return self.resultado.y_pred
        if isinstance(X, pd.Series):
            raise TypeError("X debe ser una tabla de características (DataFrame).")
        matriz = self._transformar(X)
        return pd.Series(self.modelo.predict(matriz), index=getattr(X, "index", None))

    def evaluar(self) -> ResultadoClasificacion:
        """Retorna el resultado más reciente para inspección y render."""
        if self.resultado is None:
            raise RuntimeError("Entrene primero con RF() o NR().")
        return self.resultado

    def RF(self, *args, **kwargs):
        """Entrena un Random Forest y devuelve métricas de clasificación."""
        return self.entrenar(algoritmo="RF", *args, **kwargs)

    def NR(self, *args, **kwargs):
        """Entrena un Naive Bayes gaussiano y devuelve métricas."""
        return self.entrenar(algoritmo="NR", *args, **kwargs)

    def previsualizar_particion(
        self,
        *,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = True,
        incluir_categoricas: bool = False,
        imputar: bool = True,
        estandarizar: bool = True,
    ) -> dict[str, Any]:
        """Calcula y retorna metadatos de partición sin entrenar modelo."""
        self._definir_target()
        self._definir_features()
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1 (excluido).")
        if self._tipo_target() != "clasificacion":
            raise TypeError("La variable objetivo debe ser categórica para previsualizar clasificación.")
        if self.datos.empty:
            raise ValueError("No hay datos para previsualizar.")

        objetivo = self.datos[self.target]
        columnas = list(self.features)
        if not columnas:
            raise ValueError("Debe seleccionar al menos una feature.")

        base = pd.concat([self.datos[columnas], objetivo], axis=1).copy()
        base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.target])
        if base.empty:
            raise ValueError("No hay filas válidas para la partición.")

        base = base.reset_index(drop=True)
        muestra = base[self.features]
        y = base[self.target]
        estrato = (
            y
            if stratify and y.nunique(dropna=False) > 1
            else None
        )
        indice_train, indice_test = train_test_split(
            muestra.index,
            test_size=test_size,
            random_state=random_state,
            stratify=estrato,
        )

        seleccionado = muestra.copy()
        nulos = int(seleccionado.isna().sum().sum())
        return {
            "tam_train": len(indice_train),
            "tam_test": len(indice_test),
            "duplicados": int(seleccionado.duplicated().sum()),
            "nulos": nulos,
            "nulos_porcentaje": float(
                base[self.features + [self.target]].isna().mean().mean() * 100
            ),
            "head": base.loc[indice_train].head(5),
            "tail": base.loc[indice_test].tail(5),
            "train_indices": list(indice_train),
            "test_indices": list(indice_test),
            "distribucion_train": (
                y.loc[indice_train].value_counts(dropna=False).to_dict()
            ),
            "distribucion_test": y.loc[indice_test].value_counts(dropna=False).to_dict(),
        }

    def _transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma un conjunto de entrada con el preprocesador de entrenamiento."""
        if self._preparados is None:
            raise RuntimeError(
                "No hay transformador disponible. Entrene primero para fijar la configuración."
            )
        if X is None:
            raise ValueError("X no puede ser None para predecir en nuevos datos.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._preparados.columnas_origen)
        matriz = self._preparados.transformador.transform(X[self.features])
        columnas = [
            nombre.split("__", maxsplit=1)[-1]
            for nombre in self._preparados.transformador.get_feature_names_out()
        ]
        return pd.DataFrame(
            np.asarray(matriz, dtype=float),
            index=X.index,
            columns=columnas,
        )

    @staticmethod
    def _ordenar_labels(real, predicho) -> list[Any]:
        """Calcula etiquetas estables para la matriz de confusión."""
        combinado = pd.concat([pd.Series(real), pd.Series(predicho)], ignore_index=True)
        etiquetas_unicas = list(dict.fromkeys(combinado.tolist()))
        try:
            return list(sorted(etiquetas_unicas))
        except TypeError:
            return etiquetas_unicas

    @staticmethod
    def _metricas_supervisadas(
        y_train: pd.Series,
        y_true: pd.Series,
        y_pred: pd.Series,
        matriz: pd.DataFrame,
        labels,
    ) -> dict[str, Any]:
        """Resume métricas globales y por clase para clasificación binaria o múltiple."""
        precision_por_clase = precision_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        recall_por_clase = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        f1_por_clase = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        reporte = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": {
                "global": float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "macro": float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "por_clase": {
                    str(etiqueta): {
                        "precision": float(precision_por_clase[indice]),
                        "recall": float(recall_por_clase[indice]),
                        "f1": float(f1_por_clase[indice]),
                    }
                    for indice, etiqueta in enumerate(labels)
                },
            },
            "recall": {
                "global": float(
                    recall_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "macro": float(
                    recall_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
            },
            "f1": {
                "global": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "macro": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
            },
            "reporte": reporte,
            "distribucion": {
                "train": y_train.value_counts(dropna=False).to_dict(),
                "real": y_true.value_counts(dropna=False).to_dict(),
                "predicho": y_pred.value_counts(dropna=False).to_dict(),
            },
            "matriz": matriz.to_dict(),
            "muestra": len(y_true),
        }

    def __str__(self) -> str:
        self._definir_target()
        base = (
            "Clase Clasificacion\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
        if self.exactitud is not None:
            base += f"\nExactitud: {self.exactitud:.4f}"
        return base
