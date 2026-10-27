"""Catálogo de regresión compartido por modelos, experimentos e interfaz."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .configuracion_clasificacion import ParametroModelo


@dataclass(frozen=True)
class ConfiguracionRegresion:
    """Identidad y controles de un regresor sin dependencias de Streamlit."""

    etiqueta: str
    parametros: tuple[ParametroModelo, ...]


ALGORITMOS_REGRESION = (
    "RLS",
    "RLM",
    "Lasso",
    "LassoCV",
    "Ridge",
    "RidgeCV",
    "SVR",
    "DT",
    "RF",
)
ALGORITMOS_LAB3 = ALGORITMOS_REGRESION
ALPHAS_REGRESION = (0.01, 0.1, 1.0, 10.0)
ALPHAS_LAB3 = ALPHAS_REGRESION
PLIEGUES_INTERNOS = 5

_INTERCEPTO = ParametroModelo(
    "fit_intercept", "Ajustar intercepto", "select", opciones=(True, False)
)
_ALPHA = ParametroModelo(
    "alpha",
    "Regularización (alpha)",
    "float",
    minimo=0.0001,
    maximo=1000.0,
    valor=1.0,
    paso=0.1,
)
_ALPHAS = ParametroModelo(
    "alphas",
    "Valores de alpha para validación interna",
    "alphas",
    valor=ALPHAS_REGRESION,
)
_PROFUNDIDAD = ParametroModelo(
    "max_depth",
    "Profundidad máxima (0 = sin límite)",
    "depth",
    minimo=0,
    maximo=100,
    valor=0,
)

CONFIGURACIONES_REGRESION = {
    "RLS": ConfiguracionRegresion("Regresión lineal simple", (_INTERCEPTO,)),
    "RLM": ConfiguracionRegresion("Regresión lineal múltiple", (_INTERCEPTO,)),
    "Lasso": ConfiguracionRegresion("Lasso", (_ALPHA, _INTERCEPTO)),
    "LassoCV": ConfiguracionRegresion("LassoCV", (_ALPHAS, _INTERCEPTO)),
    "Ridge": ConfiguracionRegresion("Ridge", (_ALPHA, _INTERCEPTO)),
    "RidgeCV": ConfiguracionRegresion("RidgeCV", (_ALPHAS, _INTERCEPTO)),
    "SVR": ConfiguracionRegresion(
        "SVR · vectores de soporte",
        (
            ParametroModelo(
                "C", "Regularización (C)", "float", 0.001, 1000.0, 1.0, 0.1
            ),
            ParametroModelo("epsilon", "Margen epsilon", "float", 0.0, 100.0, 0.1, 0.1),
            ParametroModelo(
                "kernel",
                "Kernel",
                "select",
                opciones=("rbf", "linear", "poly", "sigmoid"),
            ),
        ),
    ),
    "DT": ConfiguracionRegresion(
        "Árbol de decisión",
        (
            _PROFUNDIDAD,
            ParametroModelo(
                "min_samples_split", "Mínimo de muestras para dividir", "int", 2, 100, 2
            ),
        ),
    ),
    "RF": ConfiguracionRegresion(
        "Random Forest",
        (
            _PROFUNDIDAD,
            ParametroModelo(
                "n_estimators", "Número de árboles", "int", 10, 1000, 100, 10
            ),
        ),
    ),
}
NOMBRES_REGRESORES = {
    algoritmo: configuracion.etiqueta
    for algoritmo, configuracion in CONFIGURACIONES_REGRESION.items()
}

# El orden es estable y se utiliza para resolver empates en el benchmark.
_CONFIGURACIONES_EXPERIMENTO = {
    "RLS": (("estandar", {}),),
    "RLM": (("estandar", {}),),
    "Lasso": (("alpha_0_1", {"alpha": 0.1}), ("alpha_1", {"alpha": 1.0})),
    "LassoCV": (("alpha_cv", {"alphas": ALPHAS_REGRESION}),),
    "Ridge": (("alpha_0_1", {"alpha": 0.1}), ("alpha_1", {"alpha": 1.0})),
    "RidgeCV": (("alpha_cv", {"alphas": ALPHAS_REGRESION}),),
    "SVR": (
        ("C_1", {"C": 1.0, "kernel": "rbf"}),
        ("C_10", {"C": 10.0, "kernel": "rbf"}),
    ),
    "DT": (("sin_limite", {"max_depth": None}), ("profundidad_5", {"max_depth": 5})),
    "RF": (
        ("sin_limite", {"n_estimators": 100, "max_depth": None}),
        ("profundidad_5", {"n_estimators": 100, "max_depth": 5}),
    ),
}


def configuraciones_experimento_regresion() -> dict[str, list[dict[str, Any]]]:
    """Devuelve catorce candidatos independientes, con formato reproducible."""
    return {
        algoritmo: [
            {"nombre": nombre, "parametros": dict(parametros)}
            for nombre, parametros in candidatos
        ]
        for algoritmo, candidatos in _CONFIGURACIONES_EXPERIMENTO.items()
    }
