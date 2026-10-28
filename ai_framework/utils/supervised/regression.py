"""Funciones de utilidad, estimadores y cálculo de métricas para regresión."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ai_framework.data.preprocessing import PreprocesadorSupervisado
from ..results import ResultadoRegresion
from .regression_config import ALGORITMOS_REGRESION
from .regression_algorithms import (
    crear_modelo,
    metadatos_regresion,
    normalizar_algoritmo,
    parametros_regresion,
)


def metricas_regresion(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula errores en unidades originales y R² con al menos dos filas."""
    if len(y_true) < 2:
        raise ValueError("La evaluación necesita al menos dos filas para calcular R².")
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def validar_objetivo(y: pd.Series) -> None:
    """Verifica que la variable objetivo sea numérica continua y válida."""
    if (
        not pd.api.types.is_numeric_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_complex_dtype(y)
    ):
        raise TypeError("La variable objetivo debe ser numérica real.")
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(
            "La variable objetivo no admite valores nulos o no finitos."
        )
    if y.nunique() < 2:
        raise ValueError(
            "La variable objetivo debe tener variación; no ser constante."
        )


def validar_datos_regresion(base: pd.DataFrame, target: str) -> pd.DataFrame:
    """Valida los datos tabulares para regresión."""
    validar_objetivo(base[target])
    return base


def validar_predictoras(features: list[str], base: pd.DataFrame, algoritmo: str) -> None:
    """Valida los requisitos de características según el algoritmo."""
    if algoritmo not in ALGORITMOS_REGRESION:
        raise ValueError(f"Algoritmo de regresión desconocido: {algoritmo}")
    if algoritmo == "RLS":
        if len(features) != 1:
            raise ValueError(
                "RLS requiere exactamente una variable predictora numérica."
            )
        serie = base[features[0]]
        if (
            not pd.api.types.is_numeric_dtype(serie)
            or pd.api.types.is_bool_dtype(serie)
            or pd.api.types.is_complex_dtype(serie)
        ):
            raise TypeError("RLS requiere una variable predictora numérica real.")
    elif algoritmo == "RLM" and len(features) < 2:
        raise ValueError("RLM requiere al menos dos variables predictoras.")


def ajustar_regresion(
    base: pd.DataFrame,
    train: pd.Index,
    features: list[str],
    target: str,
    algoritmo: str,
    random_state: int,
    opciones: dict[str, Any],
    parametros: dict[str, Any],
):
    """Ajusta preprocesamiento y modelo de regresión en el conjunto de entrenamiento."""
    validar_predictoras(features, base, algoritmo)
    if len(train) < 2:
        raise ValueError("Entrenamiento necesita al menos dos filas.")
    y_train = base.loc[train, target]
    validar_objetivo(y_train)
    preprocesador = PreprocesadorSupervisado(
        **{**opciones, "random_state": random_state, "seleccion_k": None}
    )
    preparados = preprocesador.ajustar_transformar(
        base.loc[train, features], y_train
    )
    modelo = crear_modelo(algoritmo, random_state, parametros)
    modelo.fit(preparados.matriz, y_train)
    return modelo, preparados


def crear_resultado_regresion(
    base: pd.DataFrame,
    train: pd.Index,
    evaluacion: pd.Index,
    features: list[str],
    target: str,
    algoritmo: str,
    modelo: Any,
    preparados: Any,
    metadatos: dict[str, Any],
) -> ResultadoRegresion:
    """Construye el contenedor ResultadoRegresion con predicciones y residuos."""
    def predecir(indices):
        X = preparados.transformador.transform(base.loc[indices, features])
        return pd.Series(modelo.predict(X), index=indices, name="prediccion")

    y_train = base.loc[train, target].rename("real").copy()
    y_true = base.loc[evaluacion, target].rename("real").copy()
    y_pred_train, y_pred = predecir(train), predecir(evaluacion)
    return ResultadoRegresion(
        algoritmo=algoritmo,
        target=target,
        features=tuple(features),
        y_true=y_true,
        y_pred=y_pred,
        y_train=y_train,
        y_pred_train=y_pred_train,
        residuos=(y_true - y_pred).rename("residuo"),
        residuos_train=(y_train - y_pred_train).rename("residuo"),
        metricas=metricas_regresion(y_true, y_pred),
        metricas_train=metricas_regresion(y_train, y_pred_train),
        muestra_train=len(train),
        muestra_test=len(evaluacion),
        modelo=modelo,
        datos_preparados=preparados,
        parametros=parametros_regresion(modelo),
        metadatos={
            **metadatos,
            **metadatos_regresion(modelo),
            "columnas_transformadas": list(preparados.matriz.columns),
            "columnas_descartadas": list(preparados.columnas_descartadas),
        },
    )
