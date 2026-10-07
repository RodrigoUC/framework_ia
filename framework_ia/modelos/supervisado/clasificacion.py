"""Clasificadores del LAB02, con evaluación reproducible y sin fuga de datos."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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
from sklearn.utils.multiclass import type_of_target

from ...resultados import ResultadoClasificacion, ResultadoParticion
from .base import Supervisado
from .particiones import (
    comprobar_clases,
    dividir_indices,
    huella_datos,
    validar_particion,
)
from .preprocesamiento import PreprocesadorSupervisado

ALGORITMOS_LAB2 = ("KNN", "DT", "RF", "XGBoost", "AdaBoost")


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
    }
    try:
        return aliases[algoritmo.strip().lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"Algoritmo no soportado: {algoritmo}. Use {ALGORITMOS_LAB2} o NR."
        ) from exc


def configuraciones_lab2() -> dict[str, list[dict[str, Any]]]:
    """Configuración estándar de sklearn/XGBoost y dos variantes acotadas.

    La semilla y n_jobs=1 son controles de reproducibilidad, no optimización.
    Se devuelven estructuras nuevas en cada llamada para permitir su edición.
    """
    return {
        "KNN": [
            {"nombre": "estandar", "parametros": {}},
            {
                "nombre": "vecinos_3_distancia",
                "parametros": {"n_neighbors": 3, "weights": "distance"},
            },
            {"nombre": "vecinos_9_manhattan", "parametros": {"n_neighbors": 9, "p": 1}},
        ],
        "DT": [
            {"nombre": "estandar", "parametros": {}},
            {
                "nombre": "profundidad_4",
                "parametros": {"max_depth": 4, "min_samples_leaf": 2},
            },
            {
                "nombre": "entropia_8",
                "parametros": {"criterion": "entropy", "max_depth": 8},
            },
        ],
        "RF": [
            {"nombre": "estandar", "parametros": {}},
            {
                "nombre": "profundidad_6",
                "parametros": {"max_depth": 6, "min_samples_leaf": 2},
            },
            {
                "nombre": "200_arboles",
                "parametros": {"n_estimators": 200, "max_features": "log2"},
            },
        ],
        "XGBoost": [
            {"nombre": "estandar", "parametros": {}},
            {
                "nombre": "profundidad_3",
                "parametros": {
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                },
            },
            {
                "nombre": "submuestreo",
                "parametros": {
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "n_estimators": 150,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
            },
        ],
        "AdaBoost": [
            {"nombre": "estandar", "parametros": {}},
            {
                "nombre": "100_estimadores",
                "parametros": {"n_estimators": 100, "learning_rate": 0.5},
            },
            {
                "nombre": "200_estimadores",
                "parametros": {"n_estimators": 200, "learning_rate": 0.1},
            },
        ],
    }


class _XGBoostEtiquetas:
    """Codifica etiquetas desde train y devuelve sus valores originales.

    XGBoost exige etiquetas enteras consecutivas; el contrato público conserva
    etiquetas string, booleanas o numéricas no consecutivas, incluso multiclase.
    """

    def __init__(self, estimador):
        self.estimador = estimador
        self.codificador = LabelEncoder()

    def fit(self, X, y):
        codificadas = self.codificador.fit_transform(y)
        self.classes_ = self.codificador.classes_
        self.estimador.fit(X, codificadas)
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


def _crear_modelo(algoritmo, random_state, parametros):
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


class Clasificacion(Supervisado):
    """KNN, árbol, Random Forest, XGBoost, AdaBoost y Naive Bayes compatible."""

    def __init__(
        self, dataframe=None, ruta_datos=None, *, target=None, features=None, **kwargs
    ):
        super().__init__(
            dataframe=dataframe,
            ruta_datos=ruta_datos,
            target=target,
            features=features,
            **kwargs,
        )
        self._reiniciar_resultados()

    def _reiniciar_resultados(self):
        """No reutilizar resultados anteriores tras un intento nuevo fallido."""
        self.resultado: ResultadoClasificacion | None = None
        self.experimento = None
        self.modelo = None
        self._preparados = None
        self.exactitud = None

    def _datos_clasificacion(self):
        if self.datos.empty:
            raise ValueError("No hay datos para entrenar.")
        if not self.datos.index.is_unique:
            raise ValueError(
                "El dataset necesita índices únicos; restablezca su índice antes de partir."
            )
        if not self.datos.columns.is_unique:
            raise ValueError("El dataset necesita nombres de columna únicos.")
        self._definir_target()
        self._definir_features()
        if self.target not in self.datos:
            raise KeyError(f"La variable objetivo '{self.target}' no existe.")
        self.features = list(self.features)
        if not self.features:
            raise ValueError("Debe seleccionar al menos una variable predictora.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Las variables predictoras no deben repetirse.")
        if self.target in self.features:
            raise ValueError("El target no debe incluirse entre las features.")
        faltantes = [c for c in self.features if c not in self.datos]
        if faltantes:
            raise KeyError(f"Columnas inexistentes: {faltantes}")
        base = self.datos.loc[:, self.features + [self.target]].copy()
        base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.target])
        if base.empty:
            raise ValueError("No hay filas válidas tras depurar objetivos nulos.")
        try:
            tipo = type_of_target(base[self.target])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "La variable objetivo debe tener etiquetas de clase homogéneas."
            ) from exc
        if tipo not in ("binary", "multiclass"):
            raise TypeError(
                "La variable objetivo debe contener clases discretas; no valores continuos de regresión."
            )
        if base[self.target].nunique() < 2:
            raise ValueError("La variable objetivo debe contener al menos dos clases.")
        return base

    def _indices(self, base, *, particion, test_size, random_state, stratify):
        if particion is not None:
            validar_particion(particion, self.datos)
            indices = [
                subset.index[subset.index.isin(base.index)]
                if subset is not None
                else pd.Index([])
                for subset in (particion.train, particion.test, particion.validacion)
            ]
            if any(indice.empty for indice in indices[:2]):
                raise ValueError("Train o test quedó vacío al quitar objetivos nulos.")
            if particion.validacion is not None and indices[2].empty:
                raise ValueError("Validación quedó vacía al quitar objetivos nulos.")
            return tuple(indices)
        train, test = dividir_indices(
            base,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return train, test, pd.Index([])

    def _ajustar(self, base, train, algoritmo, random_state, opciones, parametros):
        preprocesador = PreprocesadorSupervisado(**opciones)
        preparados = preprocesador.ajustar_transformar(base.loc[train, self.features])
        modelo = _crear_modelo(algoritmo, random_state, parametros)
        modelo.fit(preparados.matriz, base.loc[train, self.target])
        return modelo, preparados

    def _resultado(
        self, base, train, evaluacion, algoritmo, modelo, preparados, metadatos
    ):
        y_train = base.loc[train, self.target]
        y_test = base.loc[evaluacion, self.target]
        X_test = preparados.transformador.transform(base.loc[evaluacion, self.features])
        y_pred = pd.Series(
            modelo.predict(X_test), name="prediccion", index=y_test.index
        )
        labels = self._ordenar_labels(y_train, y_test)
        matriz = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=pd.Index(labels, name="real"),
            columns=pd.Index(labels, name="prediccion"),
        )
        return ResultadoClasificacion(
            algoritmo=algoritmo,
            target=self.target,
            features=tuple(self.features),
            labels=tuple(labels),
            y_true=pd.Series(y_test, name="real", copy=True),
            y_pred=y_pred,
            matriz_confusion=matriz,
            metricas=self._metricas_supervisadas(
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

    def _metadatos(self, train, test, validacion, *, random_state, particion, opciones):
        return {
            "semilla": random_state,
            "semilla_particion": particion.semilla
            if particion is not None
            else random_state,
            "particion_externa": particion is not None,
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "validacion_indices": validacion.tolist(),
            "huella_datos": huella_datos(self.datos),
            "target": self.target,
            "features": list(self.features),
            "filas_objetivo_nulo": int(
                self.datos[self.target].replace([np.inf, -np.inf], np.nan).isna().sum()
            ),
            "preprocesamiento": {**opciones, "ajustado_solo_train": True},
            "versiones": {
                nombre: version(nombre)
                for nombre in ("numpy", "pandas", "scikit-learn")
            },
        }

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
        particion: ResultadoParticion | None = None,
        **kwargs,
    ):
        """Separa primero las filas y ajusta toda transformación solo con train."""
        self._reiniciar_resultados()
        algoritmo = normalizar_algoritmo(algoritmo)
        base = self._datos_clasificacion()
        train, test, validacion = self._indices(
            base,
            particion=particion,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        comprobar_clases(base.loc[train, self.target], base.loc[test, self.target])
        opciones = {
            "incluir_categoricas": incluir_categoricas,
            "imputar": imputar,
            "estandarizar": estandarizar,
        }
        modelo, preparados = self._ajustar(
            base, train, algoritmo, random_state, opciones, kwargs
        )
        metadatos = self._metadatos(
            train,
            test,
            validacion,
            random_state=random_state,
            particion=particion,
            opciones=opciones,
        )
        if algoritmo == "XGBoost":
            metadatos["versiones"]["xgboost"] = version("xgboost")
        resultado = self._resultado(
            base, train, test, algoritmo, modelo, preparados, metadatos
        )
        self.resultado, self.modelo, self._preparados = resultado, modelo, preparados
        self.exactitud = resultado.metricas["accuracy"]
        return resultado

    def experimentar(self, **kwargs):
        """Compara estándar/variantes por validación y evalúa ganadores en test."""
        self._reiniciar_resultados()
        from .experimentos import ejecutar_experimento

        self.experimento = ejecutar_experimento(self, **kwargs)
        return self.experimento

    benchmark = experimentar

    def predecir(self, X=None):
        if self.modelo is None:
            raise RuntimeError("No hay un modelo entrenado.")
        if X is None:
            return self.evaluar().y_pred.copy()
        if isinstance(X, pd.Series):
            raise TypeError("X debe ser una tabla de características (DataFrame).")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        return pd.Series(
            self.modelo.predict(self._transformar(X)), index=X.index, name="prediccion"
        )

    def _transformar(self, X):
        if self._preparados is None:
            raise RuntimeError("No hay transformador disponible; entrene primero.")
        return self._preparados.transformador.transform(X)

    def evaluar(self):
        if self.resultado is None:
            raise RuntimeError("Entrene primero un clasificador.")
        return self.resultado

    def knn(self, **kwargs):
        return self.entrenar(algoritmo="KNN", **kwargs)

    def arbol_decision(self, **kwargs):
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        return self.entrenar(algoritmo="RF", **kwargs)

    def xgboost(self, **kwargs):
        return self.entrenar(algoritmo="XGBoost", **kwargs)

    def adaboost(self, **kwargs):
        return self.entrenar(algoritmo="AdaBoost", **kwargs)

    def naive_bayes(self, **kwargs):
        return self.entrenar(algoritmo="NR", **kwargs)

    KNN = knn
    DT = dt = decision_tree = arbol_decision
    RF = rf = random_forest
    XGBoost = XGB = xgb = xgboost
    AdaBoost = ADA = ada = adaboost
    NR = nr = naive_bayes

    def previsualizar_particion(
        self,
        *,
        test_size=0.25,
        random_state=42,
        stratify=True,
        incluir_categoricas=False,
        imputar=True,
        estandarizar=True,
        particion=None,
    ):
        """Describe exactamente los índices reutilizados después al entrenar."""
        base = self._datos_clasificacion()
        train, test, validacion = self._indices(
            base,
            particion=particion,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        seleccion = base[self.features]
        return {
            "tam_train": len(train),
            "tam_test": len(test),
            "tam_validacion": len(validacion),
            "duplicados": int(seleccion.duplicated().sum()),
            "nulos": int(seleccion.isna().sum().sum()),
            "nulos_porcentaje": float(base.isna().mean().mean() * 100),
            "head": base.loc[train].head(5),
            "tail": base.loc[test].tail(5),
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "validacion_indices": validacion.tolist(),
            "distribucion_train": base.loc[train, self.target].value_counts().to_dict(),
            "distribucion_test": base.loc[test, self.target].value_counts().to_dict(),
        }

    @staticmethod
    def _ordenar_labels(real, predicho):
        etiquetas = list(
            dict.fromkeys(pd.concat([pd.Series(real), pd.Series(predicho)]).tolist())
        )
        try:
            return sorted(etiquetas)
        except TypeError:
            return etiquetas

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

    def __str__(self):
        self._definir_target()
        resumen = f"Clase Clasificacion\nTarget: {self.target}\nFeatures: {self._definir_features()}"
        return resumen + (
            f"\nExactitud: {self.exactitud:.4f}" if self.exactitud is not None else ""
        )
