"""Supervised regression models with reproducible partitions and numerical evaluation."""

from __future__ import annotations

from ai_framework.utils.supervised.regression import metricas_regresion
from .supervised import Supervised


class Regression(Supervised):
    """Numerical regression models learning preprocessing transformations only on train data."""

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

    def simple_linear_regression(self, **kwargs):
        """Fit a simple linear regression model."""
        return self.entrenar(algoritmo="RLS", **kwargs)

    def multiple_linear_regression(self, **kwargs):
        """Fit a multiple linear regression model."""
        return self.entrenar(algoritmo="RLM", **kwargs)

    def lasso(self, **kwargs):
        """Fit a regularized Lasso model."""
        return self.entrenar(algoritmo="Lasso", **kwargs)

    def lasso_cv(self, **kwargs):
        """Fit a Lasso model with internal cross-validation."""
        return self.entrenar(algoritmo="LassoCV", **kwargs)

    def ridge(self, **kwargs):
        """Fit a regularized Ridge model."""
        return self.entrenar(algoritmo="Ridge", **kwargs)

    def ridge_cv(self, **kwargs):
        """Fit a Ridge model with internal cross-validation."""
        return self.entrenar(algoritmo="RidgeCV", **kwargs)

    def svr(self, **kwargs):
        """Fit a Support Vector Regressor (SVR)."""
        return self.entrenar(algoritmo="SVR", **kwargs)

    def decision_tree(self, **kwargs):
        """Fit a regression decision tree."""
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        """Fit a random forest regressor ensemble."""
        return self.entrenar(algoritmo="RF", **kwargs)

    def __str__(self) -> str:
        self._definir_target()
        base = (
            "Class Regression\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
        if self.r2 is not None:
            base += f"\nR²: {self.r2:.4f}"
        if self.exactitud is not None:
            base += f"\nAccuracy: {self.exactitud:.4f}"
        return base


def __getattr__(name: str):
    if name == "Regresion":
        import ai_framework.utils.aliases as aliases
        return aliases.Regresion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
