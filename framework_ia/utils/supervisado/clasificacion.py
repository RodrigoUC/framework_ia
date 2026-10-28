"""Funciones de utilidad, estimadores y cálculo de métricas para clasificación."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target

from ...datos.preprocesamiento import PreprocesadorSupervisado
from ..resultados import ResultadoClasificacion
from .configuracion_clasificacion import (
    ALGORITMOS_CLASIFICACION,
    ALGORITMOS_LAB2,
    configuraciones_experimento,
)


def normalizar_algoritmo(algoritmo: str) -> str:
    """Admite los nombres de clase y aliases de la interfaz pythónica."""
    aliases = {
        "knn": "KNN",
        "k_nearest_neighbors": "KNN",
        "vecinos_cercanos": "KNN",
        "dt": "DT",
        "decision_tree": "DT",
        "arbol_decision": "DT",
        "rf": "RF",
        "random_forest": "RF",
        "randomforest": "RF",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
        "ada": "AdaBoost",
        "adaboost": "AdaBoost",
        "ada_boost": "AdaBoost",
        "nr": "NR",
        "nb": "NR",
        "naive_bayes": "NR",
        "gaussian_nb": "NR",
        "rl": "RL",
        "regresion_logistica": "RL",
        "logistic_regression": "RL",
        "logisticregression": "RL",
    }
    try:
        return aliases[algoritmo.strip().lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"Algoritmo no soportado: {algoritmo}. Use {ALGORITMOS_CLASIFICACION}, NR o RL."
        ) from exc


def configuraciones_lab2() -> dict[str, list[dict[str, Any]]]:
    """Catálogo de configuraciones estándar para experimentación de clasificación."""
    return configuraciones_experimento()


class _XGBoostEtiquetas:
    """Codifica etiquetas desde train y devuelve sus valores originales.

    XGBoost exige etiquetas enteras consecutivas; el contrato público conserva
    etiquetas string, booleanas o numéricas no consecutivas, incluso multiclase.
    """

    def __init__(self, estimador):
        self.estimador = estimador
        self.codificador = LabelEncoder()

    def fit(self, X, y, sample_weight=None):
        codificadas = self.codificador.fit_transform(y)
        self.classes_ = self.codificador.classes_
        self.estimador.fit(X, codificadas, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.codificador.inverse_transform(self.estimador.predict(X).astype(int))

    def predict_proba(self, X):
        return self.estimador.predict_proba(X)

    def get_params(self, deep=True):
        return self.estimador.get_params(deep=deep)

    @property
    def feature_importances_(self):
        return self.estimador.feature_importances_


def crear_modelo(algoritmo: str, random_state: int, parametros: dict):
    """Instancia el estimador de clasificación correspondiente según el algoritmo."""
    parametros = dict(parametros)
    if algoritmo == "KNN":
        return KNeighborsClassifier(**{"n_jobs": 1, **parametros})
    if algoritmo == "DT":
        return DecisionTreeClassifier(**{"random_state": random_state, **parametros})
    if algoritmo == "RF":
        return RandomForestClassifier(
            **{"random_state": random_state, "n_jobs": 1, **parametros}
        )
    if algoritmo == "AdaBoost":
        return AdaBoostClassifier(**{"random_state": random_state, **parametros})
    if algoritmo == "NR":
        return GaussianNB(**parametros)
    if algoritmo == "RL":
        return LogisticRegression(
            **{"random_state": random_state, "max_iter": 1000, **parametros}
        )
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost no está instalado. Instale requirements.txt para ejecutar este algoritmo."
        ) from exc
    return _XGBoostEtiquetas(
        XGBClassifier(
            **{
                "random_state": random_state,
                "n_jobs": 1,
                **parametros,
            }
        )
    )


def validar_datos_clasificacion(base: pd.DataFrame, target: str) -> pd.DataFrame:
    """Valida que los datos sean aptos para clasificación supervisada."""
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if base.empty:
        raise ValueError("No hay filas válidas tras depurar objetivos nulos.")
    try:
        tipo = type_of_target(base[target])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "La variable objetivo debe tener etiquetas de clase homogéneas."
        ) from exc
    if tipo not in ("binary", "multiclass"):
        raise TypeError(
            "La variable objetivo debe contener clases discretas; no valores continuos de regresión."
        )
    if base[target].nunique() < 2:
        raise ValueError("La variable objetivo debe contener al menos dos clases.")
    return base


def ajustar_clasificacion(
    base: pd.DataFrame,
    train: pd.Index,
    features: list[str],
    target: str,
    algoritmo: str,
    random_state: int,
    opciones: dict[str, Any],
    parametros: dict[str, Any],
):
    """Ajusta preprocesamiento y clasificador sólo sobre los datos de entrenamiento."""
    parametros = dict(parametros)
    balancear = bool(parametros.pop("balancear_clases", False))
    y_train = base.loc[train, target]
    preprocesador = PreprocesadorSupervisado(
        **{"random_state": random_state, **opciones}
    )
    preparados = preprocesador.ajustar_transformar(
        base.loc[train, features], y_train
    )
    modelo = crear_modelo(algoritmo, random_state, parametros)
    if balancear:
        if algoritmo == "KNN":
            raise ValueError("KNN no admite pesos por observación.")
        modelo.fit(
            preparados.matriz,
            y_train,
            sample_weight=compute_sample_weight("balanced", y_train),
        )
    else:
        modelo.fit(preparados.matriz, y_train)
    return modelo, preparados


def ordenar_labels(real: pd.Series, predicho: pd.Series):
    """Genera orden determinista para las etiquetas de clase."""
    etiquetas = list(
        dict.fromkeys(pd.concat([pd.Series(real), pd.Series(predicho)]).tolist())
    )
    try:
        return sorted(etiquetas)
    except TypeError:
        return etiquetas


def metricas_clasificacion(
    y_train: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    matriz: pd.DataFrame,
    labels: list[Any],
) -> dict[str, Any]:
    """Calcula métricas globales y por clase para clasificación binaria o multiclase."""
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
                    labels=labels,
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
                    labels=labels,
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
                f1_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                )
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


def crear_resultado_clasificacion(
    base: pd.DataFrame,
    train: pd.Index,
    evaluacion: pd.Index,
    features: list[str],
    target: str,
    algoritmo: str,
    modelo: Any,
    preparados: Any,
    metadatos: dict[str, Any],
) -> ResultadoClasificacion:
    """Construye la estructura de resultado de clasificación."""
    y_train = base.loc[train, target]
    y_test = base.loc[evaluacion, target]
    X_test = preparados.transformador.transform(base.loc[evaluacion, features])
    y_pred = pd.Series(
        modelo.predict(X_test), name="prediccion", index=y_test.index
    )
    labels = ordenar_labels(y_train, y_test)
    matriz = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=pd.Index(labels, name="real"),
        columns=pd.Index(labels, name="prediccion"),
    )
    return ResultadoClasificacion(
        algoritmo=algoritmo,
        target=target,
        features=tuple(features),
        labels=tuple(labels),
        y_true=pd.Series(y_test, name="real", copy=True),
        y_pred=y_pred,
        matriz_confusion=matriz,
        metricas=metricas_clasificacion(
            y_train, y_test, y_pred, matriz, labels
        ),
        muestra_train=len(train),
        muestra_test=len(evaluacion),
        modelo=modelo,
        parametros=modelo.get_params(),
        metadatos={
            **metadatos,
            "columnas_transformadas": list(preparados.matriz.columns),
            "columnas_descartadas": list(preparados.columnas_descartadas),
        },
    )
