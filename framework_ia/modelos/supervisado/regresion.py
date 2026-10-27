"""Regresión supervisada con particiones reproducibles y evaluación numérica."""

from __future__ import annotations

from typing import Any

from ...utils.supervisado import (
    ajustar_regresion,
    crear_resultado_regresion,
    ejecutar_experimento_regresion,
    metricas_regresion,
    normalizar_algoritmo_regresion,
    validar_datos_regresion,
    validar_objetivo_regresion,
    validar_predictoras_regresion,
)
from .supervisado import Supervisado


class Regresion(Supervisado):
    """Modelos numéricos de regresión que aprenden transformaciones sólo de train."""

    _mensaje_sin_entrenar = "Entrene primero un regresor."

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
        return False

    def _normalizar_algoritmo(self, algoritmo: str | None) -> str:
        alg = str(algoritmo or "RLM").strip()
        if alg.lower() in {
            "rl",
            "regresion_logistica",
            "logistic_regression",
            "logisticregression",
        }:
            from .clasificacion import Clasificacion

            return Clasificacion(
                dataframe=self.datos, target=self.target, features=self.features
            ).RL

        return normalizar_algoritmo_regresion(alg)

    def _datos_regresion(self):
        base = self._validar_estructura_base(
            mensaje_indices="El dataset necesita índices únicos."
        )
        return validar_datos_regresion(base, self.target)

    _datos_supervisados = _datos_regresion

    _validar_objetivo = staticmethod(validar_objetivo_regresion)

    def _validar_predictoras(self, base, algoritmo):
        return validar_predictoras_regresion(self.features, base, algoritmo)

    def _validar_entrenamiento(self, base, train, test, validacion) -> None:
        if len(test) < 2:
            raise ValueError(
                "La evaluación necesita al menos dos filas para calcular R²."
            )

    def _ajustar(self, base, train, algoritmo, random_state, opciones, parametros):
        return ajustar_regresion(
            base, train, self.features, self.target, algoritmo, random_state, opciones, parametros
        )

    def _resultado(self, base, train, evaluacion, algoritmo, modelo, preparados, metadatos):
        return crear_resultado_regresion(
            base, train, evaluacion, self.features, self.target, algoritmo, modelo, preparados, metadatos
        )

    def _actualizar_metricas_resumen(self, resultado) -> None:
        self.r2 = resultado.metricas["r2"]

    def _ejecutar_experimento(self, **kwargs):
        intento = ejecutar_experimento_regresion(self, **kwargs)
        elegido = intento.mejores.get(intento.mejor_algoritmo)
        if elegido is not None:
            self.experimento = intento
            self.resultado = elegido
            self.modelo = elegido.modelo
            self._preparados = elegido.datos_preparados
            self.r2 = elegido.metricas["r2"]
        return intento

    def lineal_simple(self, **kwargs):
        return self.entrenar(algoritmo="RLS", **kwargs)

    def lineal_multiple(self, **kwargs):
        return self.entrenar(algoritmo="RLM", **kwargs)

    def lasso(self, **kwargs):
        return self.entrenar(algoritmo="Lasso", **kwargs)

    def lasso_cv(self, **kwargs):
        return self.entrenar(algoritmo="LassoCV", **kwargs)

    def ridge(self, **kwargs):
        return self.entrenar(algoritmo="Ridge", **kwargs)

    def ridge_cv(self, **kwargs):
        return self.entrenar(algoritmo="RidgeCV", **kwargs)

    def svr(self, **kwargs):
        return self.entrenar(algoritmo="SVR", **kwargs)

    def arbol_decision(self, **kwargs):
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        return self.entrenar(algoritmo="RF", **kwargs)

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
