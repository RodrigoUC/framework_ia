"""Clasificadores supervisados con evaluación reproducible y contratos pythónicos."""

from __future__ import annotations

from ...utils.supervisado.clasificacion import configuraciones_lab2
from .supervisado import Supervisado


class Clasificacion(Supervisado):
    """Clasificación supervisada: KNN, árboles, ensambles, Naive Bayes y regresión logística."""

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

    def knn(self, **kwargs):
        """Ajusta un clasificador K-Nearest Neighbors."""
        return self.entrenar(algoritmo="KNN", **kwargs)

    def arbol_decision(self, **kwargs):
        """Ajusta un clasificador de árbol de decisión."""
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        """Ajusta un ensamble de bosques aleatorios para clasificación."""
        return self.entrenar(algoritmo="RF", **kwargs)

    def xgboost(self, **kwargs):
        """Ajusta un clasificador XGBoost."""
        return self.entrenar(algoritmo="XGBoost", **kwargs)

    def adaboost(self, **kwargs):
        """Ajusta un ensamble AdaBoost."""
        return self.entrenar(algoritmo="AdaBoost", **kwargs)

    def naive_bayes(self, **kwargs):
        """Ajusta un clasificador Naive Bayes."""
        return self.entrenar(algoritmo="NR", **kwargs)

    def regresion_logistica(self, **kwargs):
        """Ajusta un modelo de regresión logística."""
        return self.entrenar(algoritmo="RL", **kwargs)

    # Alias pythonicos
    KNN = knn
    DT = dt = decision_tree = arbol_decision
    RF = rf = random_forest
    XGBoost = XGB = xgb = xgboost
    AdaBoost = ADA = ada = adaboost
    NR = nr = naive_bayes
    RL = rl = logistic_regression = regresion_logistica

    def __str__(self) -> str:
        self._definir_target()
        resumen = f"Clase Clasificacion\nTarget: {self.target}\nFeatures: {self._definir_features()}"
        if self.exactitud is not None:
            resumen += f"\nExactitud: {self.exactitud:.4f}"
        return resumen
