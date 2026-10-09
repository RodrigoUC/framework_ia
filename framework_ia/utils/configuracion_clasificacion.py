"""Configuración compartida para clasificación supervisada."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParametroModelo:
    """Metadatos de un control de parámetros expuesto por la interfaz."""

    nombre: str
    etiqueta: str
    tipo: str
    minimo: int | float | None = None
    maximo: int | float | None = None
    valor: Any = None
    paso: int | float | None = None
    opciones: tuple[Any, ...] = ()
    formato: str | None = None


@dataclass(frozen=True)
class ConfiguracionModelo:
    """Identidad y parámetros configurables de un clasificador de LAB02."""

    etiqueta: str
    parametros: tuple[ParametroModelo, ...]


ALGORITMOS_LAB2 = ("KNN", "DT", "RF", "XGBoost", "AdaBoost")
METRICAS_SELECCION = {
    "f1_macro": "F1 macro",
    "accuracy": "Accuracy",
    "precision_macro": "Precisión macro",
    "recall_macro": "Recall macro",
}


def _p(
    nombre,
    etiqueta,
    tipo,
    minimo=None,
    maximo=None,
    valor=None,
    paso=None,
    opciones=(),
    formato=None,
):
    return ParametroModelo(
        nombre, etiqueta, tipo, minimo, maximo, valor, paso, opciones, formato
    )


CONFIGURACIONES_MODELO = {
    "KNN": ConfiguracionModelo(
        "KNN · vecinos cercanos",
        (
            _p("n_neighbors", "Vecinos (k)", "int", 1, 99, 5),
            _p("weights", "Pesos", "select", opciones=("uniform", "distance")),
            _p(
                "p",
                "Distancia de Minkowski",
                "select",
                opciones=(2, 1),
                formato="distance",
            ),
        ),
    ),
    "DT": ConfiguracionModelo(
        "DT · árbol de decisión",
        (
            _p("max_depth", "Profundidad máxima (0 = sin límite)", "depth", 0, 100, 0),
            _p(
                "min_samples_split", "Mínimo de muestras para dividir", "int", 2, 100, 2
            ),
            _p(
                "criterion",
                "Criterio",
                "select",
                opciones=("gini", "entropy", "log_loss"),
            ),
        ),
    ),
    "RF": ConfiguracionModelo(
        "Random Forest",
        (
            _p("max_depth", "Profundidad máxima (0 = sin límite)", "depth", 0, 100, 0),
            _p(
                "min_samples_split", "Mínimo de muestras para dividir", "int", 2, 100, 2
            ),
            _p("n_estimators", "Número de árboles", "int", 10, 1000, 100, 10),
        ),
    ),
    "XGBoost": ConfiguracionModelo(
        "XGBoost",
        (
            _p("n_estimators", "Número de estimadores", "int", 10, 1000, 100, 10),
            _p("learning_rate", "Tasa de aprendizaje", "float", 0.01, 2.0, 0.3, 0.01),
            _p("max_depth", "Profundidad máxima", "int", 1, 30, 6),
        ),
    ),
    "AdaBoost": ConfiguracionModelo(
        "AdaBoost",
        (
            _p("n_estimators", "Número de estimadores", "int", 10, 1000, 50, 10),
            _p("learning_rate", "Tasa de aprendizaje", "float", 0.01, 2.0, 1.0, 0.01),
        ),
    ),
}
NOMBRES_MODELOS = {
    algoritmo: configuracion.etiqueta
    for algoritmo, configuracion in CONFIGURACIONES_MODELO.items()
}

_CONFIGURACIONES_EXPERIMENTO = {
    "KNN": (
        ("estandar", {}),
        ("vecinos_3_distancia", {"n_neighbors": 3, "weights": "distance"}),
        ("vecinos_9_manhattan", {"n_neighbors": 9, "p": 1}),
    ),
    "DT": (
        ("estandar", {}),
        ("profundidad_4", {"max_depth": 4, "min_samples_leaf": 2}),
        ("entropia_8", {"criterion": "entropy", "max_depth": 8}),
    ),
    "RF": (
        ("estandar", {}),
        ("profundidad_6", {"max_depth": 6, "min_samples_leaf": 2}),
        ("200_arboles", {"n_estimators": 200, "max_features": "log2"}),
    ),
    "XGBoost": (
        ("estandar", {}),
        ("profundidad_3", {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}),
        (
            "submuestreo",
            {
                "max_depth": 4,
                "learning_rate": 0.05,
                "n_estimators": 150,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ),
    ),
    "AdaBoost": (
        ("estandar", {}),
        ("100_estimadores", {"n_estimators": 100, "learning_rate": 0.5}),
        ("200_estimadores", {"n_estimators": 200, "learning_rate": 0.1}),
    ),
}


def configuraciones_experimento() -> dict[str, list[dict[str, Any]]]:
    """Return fresh experiment candidate configurations for backward compatibility."""
    return {
        algoritmo: [
            {"nombre": nombre, "parametros": dict(parametros)}
            for nombre, parametros in configuraciones
        ]
        for algoritmo, configuraciones in _CONFIGURACIONES_EXPERIMENTO.items()
    }
