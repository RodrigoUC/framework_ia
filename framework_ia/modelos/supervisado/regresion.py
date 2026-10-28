"""Regresión supervisada con particiones reproducibles y evaluación numérica."""

from __future__ import annotations

from ...utils.supervisado.regresion import metricas_regresion
from .supervisado import Supervisado


class Regresion(Supervisado):
    """Modelos numéricos de regresión que aprenden transformaciones sólo de train."""

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

    def lineal_simple(self, **kwargs):
        """Ajusta un modelo de regresión lineal simple."""
        return self.entrenar(algoritmo="RLS", **kwargs)

    def lineal_multiple(self, **kwargs):
        """Ajusta un modelo de regresión lineal múltiple."""
        return self.entrenar(algoritmo="RLM", **kwargs)

    def lasso(self, **kwargs):
        """Ajusta un modelo regularizado Lasso."""
        return self.entrenar(algoritmo="Lasso", **kwargs)

    def lasso_cv(self, **kwargs):
        """Ajusta un modelo Lasso con validación cruzada interna."""
        return self.entrenar(algoritmo="LassoCV", **kwargs)

    def ridge(self, **kwargs):
        """Ajusta un modelo regularizado Ridge."""
        return self.entrenar(algoritmo="Ridge", **kwargs)

    def ridge_cv(self, **kwargs):
        """Ajusta un modelo Ridge con validación cruzada interna."""
        return self.entrenar(algoritmo="RidgeCV", **kwargs)

    def svr(self, **kwargs):
        """Ajusta una máquina de vectores de soporte para regresión (SVR)."""
        return self.entrenar(algoritmo="SVR", **kwargs)

    def arbol_decision(self, **kwargs):
        """Ajusta un árbol de decisión para regresión."""
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        """Ajusta un ensamble de bosques aleatorios para regresión."""
        return self.entrenar(algoritmo="RF", **kwargs)

    # Alias pythonicos
    RLS = lineal_simple
    RLM = lineal_multiple
    Lasso = lasso
    LassoCV = lasso_cv
    Ridge = ridge
    RidgeCV = ridge_cv
    SVR = svr
    DT = dt = decision_tree = arbol_decision
    RF = rf = random_forest

    def __str__(self) -> str:
        self._definir_target()
        base = (
            "Clase Regresion\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
        if self.r2 is not None:
            base += f"\nR²: {self.r2:.4f}"
        if self.exactitud is not None:
            base += f"\nExactitud: {self.exactitud:.4f}"
        return base
