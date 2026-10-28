"""Construcción de estimadores nativos de regresión y registro de sus parámetros efectivos."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .configuracion_regresion import (
    ALGORITMOS_LAB3,
    ALPHAS_LAB3,
    PLIEGUES_INTERNOS,
)

_ESTIMADORES = {
    "RLS": LinearRegression,
    "RLM": LinearRegression,
    "Lasso": Lasso,
    "LassoCV": LassoCV,
    "Ridge": Ridge,
    "RidgeCV": RidgeCV,
    "SVR": SVR,
    "DT": DecisionTreeRegressor,
    "RF": RandomForestRegressor,
}
_ALIASES = {nombre.lower(): nombre for nombre in ALGORITMOS_LAB3}
_ALIASES.update(
    {
        "lineal_simple": "RLS",
        "lineal_multiple": "RLM",
        "lasso_cv": "LassoCV",
        "ridge_cv": "RidgeCV",
        "arbol_decision": "DT",
        "decision_tree": "DT",
        "decisiontreeregressor": "DT",
        "random_forest": "RF",
        "randomforestregressor": "RF",
    }
)


def normalizar_algoritmo(algoritmo: str) -> str:
    """Acepta identificadores del catálogo y métodos pythonicos."""
    try:
        return _ALIASES[algoritmo.strip().lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Algoritmo de regresión desconocido: {algoritmo}") from exc


def crear_modelo(algoritmo: str, random_state: int, parametros: dict):
    """Construye un estimador nuevo; CV interna usa cinco pliegues reproducibles."""
    algoritmo = normalizar_algoritmo(algoritmo)
    opciones = dict(parametros)
    if algoritmo in ("LassoCV", "RidgeCV"):
        if "cv" in opciones or "scoring" in opciones:
            raise ValueError(
                "La validación interna fija cv en cinco pliegues y su scoring; no admite overrides."
            )
        opciones.setdefault("alphas", ALPHAS_LAB3)
        opciones["cv"] = KFold(
            n_splits=PLIEGUES_INTERNOS, shuffle=True, random_state=random_state
        )
        if algoritmo == "RidgeCV":
            opciones["scoring"] = "neg_mean_squared_error"
    if algoritmo in ("Lasso", "LassoCV", "Ridge", "DT", "RF"):
        opciones["random_state"] = random_state
    if algoritmo in ("LassoCV", "RF"):
        opciones.setdefault("n_jobs", 1)
    return _ESTIMADORES[algoritmo](**opciones)


def parametros_regresion(modelo) -> dict:
    """Describe KFold y arrays de alpha sin objetos ni direcciones de memoria."""
    parametros = modelo.get_params()
    cv = parametros.get("cv")
    if isinstance(cv, KFold):
        parametros["cv"] = {
            "tipo": "KFold",
            "n_splits": cv.n_splits,
            "shuffle": cv.shuffle,
            "random_state": cv.random_state,
        }
    if isinstance(parametros.get("alphas"), np.ndarray):
        parametros["alphas"] = parametros["alphas"].tolist()
    return parametros


def metadatos_regresion(modelo) -> dict:
    """Explicita la selección interna sin confundirla con validación externa."""
    if not isinstance(modelo, (LassoCV, RidgeCV)):
        return {}
    return {
        "alpha_elegido": float(modelo.alpha_),
        "cv_interna": {
            **parametros_regresion(modelo)["cv"],
            "scoring": "mean_squared_error"
            if isinstance(modelo, LassoCV)
            else "neg_mean_squared_error",
            "preprocesamiento_por_pliegue": False,
            "uso": "seleccion_alpha_solo_train",
        },
    }
