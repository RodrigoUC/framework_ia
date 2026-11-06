"""Regression tools built on :class:`ai_framework.models.supervised.Regression`.

They return metrics and aggregated diagnostics; predictions per row never
reach the LLM.
"""

from __future__ import annotations

from ai_framework.models.supervised import Regression
from ai_framework.utils.results import ResultadoExperimentoRegresion
from ai_framework.utils.supervised.regression_config import ALGORITMOS_REGRESION
from ai_framework.agents.tools.registry import tool
from ai_framework.agents.tools.supervised_tools import (
    MAX_ROWS,
    model_importance,
    round4,
    training_options,
)


def _metrics(metrics: dict) -> dict:
    return {key: round4(metrics[key]) for key in ("rmse", "mae", "r2") if key in metrics}


def summarize_regression(experiment: ResultadoExperimentoRegresion, n_features: int) -> dict:
    """Winner by validation RMSE, its test metrics and the candidates' outcome."""
    best = experiment.mejores.get(experiment.mejor_algoritmo)
    if best is None:
        raise ValueError("Ningún regresor pudo evaluarse; revise los errores de los candidatos.")
    return {
        "selection_metric": "rmse",
        "best_algorithm": experiment.mejor_algoritmo,
        "best_configuration": best.metadatos.get("configuracion"),
        "validation_rmse": round4(best.metadatos.get("puntuacion_validacion")),
        "validation_rmse_std": round4(best.metadatos.get("puntuacion_validacion_std")),
        "test": _metrics(best.metricas),
        "train": _metrics(best.metricas_train),
        "train_rows": int(best.muestra_train),
        "test_rows": int(best.muestra_test),
        "features": n_features,
        "candidates_ok": int(experiment.metadatos.get("candidatos_exitosos", 0)),
        "candidates_failed": int(experiment.metadatos.get("candidatos_fallidos", 0)),
    }


def _best(blackboard):
    experiment = blackboard.results["regression_experiment"]
    return experiment.mejores[experiment.mejor_algoritmo]


@tool(
    "Ranking de todos los regresores evaluados por RMSE de validación cruzada "
    "(menor es mejor), con R² de validación y estado de cada candidato."
)
def compare_regressors(blackboard) -> dict:
    table = blackboard.results["regression_experiment"].tabla
    table = table.sort_values("rmse_validacion", na_position="last").head(MAX_ROWS)
    return {
        "rows": [
            {
                "algorithm": row["algoritmo"],
                "configuration": row["configuracion"],
                "validation_rmse": round4(row["rmse_validacion"]),
                "validation_rmse_std": round4(row["puntuacion_validacion_std"]),
                "validation_r2": round4(row["r2_validacion"]),
                "selected": bool(row["seleccionado"]),
                "status": row["estado"],
                "error": row["error"],
            }
            for _, row in table.iterrows()
        ]
    }


@tool(
    "Entrena un regresor con su configuración estándar y la misma partición, "
    "y devuelve sus métricas de entrenamiento y de prueba.",
    algorithm=f"Uno de {ALGORITMOS_REGRESION}.",
)
def train_regressor(blackboard, algorithm: str) -> dict:
    if algorithm not in ALGORITMOS_REGRESION:
        raise ValueError(f"Algoritmo no soportado {algorithm!r}; use {ALGORITMOS_REGRESION}.")
    options = training_options(blackboard)
    features = blackboard.results["supervised_features"]
    if algorithm == "RLS":
        features = [blackboard.results["simple_feature"]]
    model = Regression(dataframe=blackboard.datos, target=blackboard.target, features=features)
    result = model.entrenar(algoritmo=algorithm, **options)
    return {"algorithm": algorithm, "test": _metrics(result.metricas), "train": _metrics(result.metricas_train)}


@tool(
    "Importancia de las variables del mejor regresor: coeficientes estandarizados "
    "en modelos lineales o importancias en árboles.",
    top="Cantidad máxima de variables a devolver.",
)
def regression_feature_importance(blackboard, top: int = 10) -> dict:
    best = _best(blackboard)
    rows = model_importance(best.modelo, best.datos_preparados.matriz.columns)
    if rows is None:
        return {"algorithm": best.algoritmo, "error": "Este modelo no expone importancias."}
    return {"algorithm": best.algoritmo, "features": rows[:top]}


@tool("Resumen de los residuos del mejor regresor en prueba: sesgo, dispersión y extremos.")
def residual_summary(blackboard) -> dict:
    best = _best(blackboard)
    residuals = best.residuos
    spread = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    return {
        "algorithm": best.algoritmo,
        "mean": round4(residuals.mean()),
        "std": round4(spread),
        "max_abs": round4(residuals.abs().max()),
        "share_beyond_2_std": round4((residuals.abs() > 2 * spread).mean()) if spread else 0.0,
        "correlation_with_prediction": round4(residuals.corr(best.y_pred)) if spread else None,
    }


REGRESSION_TOOLS = (compare_regressors, train_regressor, regression_feature_importance, residual_summary)
