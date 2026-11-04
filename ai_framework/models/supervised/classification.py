"""Supervised classification models with reproducible evaluation and pythonic contracts."""

from __future__ import annotations

from ai_framework.utils.supervised.classification import configuraciones_lab2
from .supervised import Supervised


class Classification(Supervised):
    """Supervised classification: KNN, decision trees, ensembles, Naive Bayes, and logistic regression."""

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
        """Fit a K-Nearest Neighbors classifier."""
        return self.entrenar(algoritmo="KNN", **kwargs)

    def decision_tree(self, **kwargs):
        """Fit a Decision Tree classifier."""
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        """Fit a Random Forest ensemble classifier."""
        return self.entrenar(algoritmo="RF", **kwargs)

    def xgboost(self, **kwargs):
        """Fit an XGBoost classifier."""
        return self.entrenar(algoritmo="XGBoost", **kwargs)

    def adaboost(self, **kwargs):
        """Fit an AdaBoost ensemble classifier."""
        return self.entrenar(algoritmo="AdaBoost", **kwargs)

    def naive_bayes(self, **kwargs):
        """Fit a Gaussian Naive Bayes classifier."""
        return self.entrenar(algoritmo="NR", **kwargs)

    def logistic_regression(self, **kwargs):
        """Fit a Logistic Regression model."""
        return self.entrenar(algoritmo="RL", **kwargs)

    def __str__(self) -> str:
        self._definir_target()
        resumen = f"Class Classification\nTarget: {self.target}\nFeatures: {self._definir_features()}"
        if self.exactitud is not None:
            resumen += f"\nAccuracy: {self.exactitud:.4f}"
        return resumen


def __getattr__(name: str):
    if name == "Clasificacion":
        import ai_framework.utils.aliases as aliases
        return aliases.Clasificacion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
