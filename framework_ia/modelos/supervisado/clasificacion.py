"""Clasificadores supervisados con evaluación reproducible y contratos pythónicos."""

from __future__ import annotations

from typing import Any

from ...datos.particion import comprobar_clases
from ...utils.supervisado import (
    ALGORITMOS_LAB2,
    ajustar_clasificacion,
    configuraciones_lab2,
    crear_resultado_clasificacion,
    ejecutar_experimento_clasificacion,
    metricas_clasificacion,
    normalizar_algoritmo_clasificacion,
    ordenar_labels,
    validar_datos_clasificacion,
)
from .supervisado import Supervisado


class Clasificacion(Supervisado):
    """Clasificación supervisada: KNN, árboles, ensambles, Naive Bayes y regresión logística."""

    _mensaje_sin_entrenar = "Entrene primero un clasificador."

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

    def _stratify_defecto(self) -> bool:
        return True

    def _normalizar_algoritmo(self, algoritmo: str | None) -> str:
        return normalizar_algoritmo_clasificacion(algoritmo or "RF")

    def _datos_clasificacion(self):
        base = self._validar_estructura_base(
            mensaje_indices="El dataset necesita índices únicos; restablezca su índice antes de partir."
        )
        return validar_datos_clasificacion(base, self.target)

    _datos_supervisados = _datos_clasificacion

    def _validar_entrenamiento(self, base, train, test, validacion) -> None:
        comprobar_clases(base.loc[train, self.target], base.loc[test, self.target])

    def _ajustar(self, base, train, algoritmo, random_state, opciones, parametros):
        return ajustar_clasificacion(
            base, train, self.features, self.target, algoritmo, random_state, opciones, parametros
        )

    def _resultado(self, base, train, evaluacion, algoritmo, modelo, preparados, metadatos):
        return crear_resultado_clasificacion(
            base, train, evaluacion, self.features, self.target, algoritmo, modelo, preparados, metadatos
        )

    def _actualizar_metricas_resumen(self, resultado) -> None:
        self.exactitud = resultado.metricas["accuracy"]

    def _ejecutar_experimento(self, **kwargs):
        self._reiniciar_resultados()
        intento = ejecutar_experimento_clasificacion(self, **kwargs)
        self.experimento = intento
        return intento

    _ordenar_labels = staticmethod(ordenar_labels)
    _metricas_supervisadas = staticmethod(metricas_clasificacion)

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

    def regresion_logistica(self, **kwargs):
        return self.entrenar(algoritmo="RL", **kwargs)

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
