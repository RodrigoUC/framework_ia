"""Classification tools built on :class:`ai_framework.models.supervised.Classification`.

They return metrics and aggregated diagnostics; predictions per row never
reach the LLM.
"""

from __future__ import annotations

from ai_framework.models.supervised import Classification
from ai_framework.utils.results import ResultadoExperimentoClasificacion
from ai_framework.utils.supervised.classification_config import ALGORITMOS_CLASIFICACION
from ai_framework.agents.tools.registry import tool
from ai_framework.agents.tools.supervised_tools import MAX_ROWS, round4, training_options


def _test_metrics(metrics: dict) -> dict:
    return {
        "accuracy": round4(metrics["accuracy"]),
        "f1_macro": round4(metrics["f1"]["macro"]),
        "precision_macro": round4(metrics["precision"]["macro"]),
        "recall_macro": round4(metrics["recall"]["macro"]),
    }


def _per_class(metrics: dict) -> dict:
    report = metrics.get("reporte", {})
    return {
        label: {
            **{key: round4(value) for key, value in scores.items()},
            "support": int(report.get(label, {}).get("support", 0)),
        }
        for label, scores in metrics["precision"]["por_clase"].items()
    }


def summarize_classification(experiment: ResultadoExperimentoClasificacion, n_features: int) -> dict:
    """Winner by validation F1 macro, its test metrics, per-class scores and baseline."""
    best = experiment.mejores.get(experiment.mejor_algoritmo)
    if best is None:
        raise ValueError("Ningún clasificador pudo evaluarse; revise los errores de los candidatos.")
    analysis = best.metadatos.get("analisis", {})
    train_counts = best.metricas["distribucion"]["train"]
    train = analysis.get("metricas_train", {})
    return {
        "selection_metric": "f1_macro",
        "best_algorithm": experiment.mejor_algoritmo,
        "best_configuration": best.metadatos.get("configuracion"),
        "validation_f1_macro": round4(best.metadatos.get("puntuacion_validacion")),
        "validation_f1_macro_std": round4(best.metadatos.get("puntuacion_validacion_std")),
        "test": _test_metrics(best.metricas),
        "train": {key: round4(train[key]) for key in ("accuracy", "f1_macro") if key in train},
        "classes": _per_class(best.metricas),
        "majority_share": round4(max(train_counts.values()) / sum(train_counts.values())),
        "train_rows": int(best.muestra_train),
        "test_rows": int(best.muestra_test),
        "features": n_features,
        "candidates_ok": int(experiment.metadatos.get("candidatos_exitosos", 0)),
        "candidates_failed": int(experiment.metadatos.get("candidatos_fallidos", 0)),
    }


def _best(blackboard):
    experiment = blackboard.results["classification_experiment"]
    return experiment.mejores[experiment.mejor_algoritmo]


@tool(
    "Ranking de todos los clasificadores evaluados por F1 macro de validación cruzada "
    "(mayor es mejor), con accuracy de validación y estado de cada candidato."
)
def compare_classifiers(blackboard) -> dict:
    table = blackboard.results["classification_experiment"].tabla
    if "etapa" in table:
        table = table[table["etapa"] == "configuracion"]
    table = table.sort_values("f1_macro_validacion", ascending=False, na_position="last").head(MAX_ROWS)
    return {
        "rows": [
            {
                "algorithm": row["algoritmo"],
                "configuration": row["configuracion"],
                "validation_f1_macro": round4(row["f1_macro_validacion"]),
                "validation_f1_macro_std": round4(row.get("puntuacion_validacion_std")),
                "validation_accuracy": round4(row["accuracy_validacion"]),
                "selected": bool(row["seleccionado"]),
                "status": row["estado"],
                "error": row["error"],
            }
            for _, row in table.iterrows()
        ]
    }


@tool(
    "Entrena un clasificador con su configuración estándar y la misma partición, "
    "y devuelve sus métricas de prueba.",
    algorithm=f"Uno de {ALGORITMOS_CLASIFICACION}.",
)
def train_classifier(blackboard, algorithm: str) -> dict:
    if algorithm not in ALGORITMOS_CLASIFICACION:
        raise ValueError(f"Algoritmo no soportado {algorithm!r}; use {ALGORITMOS_CLASIFICACION}.")
    model = Classification(
        dataframe=blackboard.datos,
        target=blackboard.target,
        features=blackboard.results["supervised_features"],
    )
    result = model.entrenar(algoritmo=algorithm, **training_options(blackboard))
    return {"algorithm": algorithm, "test": _test_metrics(result.metricas)}


@tool(
    "Importancia por permutación de las variables del mejor clasificador: cuánto cae "
    "el F1 macro de prueba al desordenar cada variable.",
    top="Cantidad máxima de variables a devolver.",
)
def classification_feature_importance(blackboard, top: int = 10) -> dict:
    best = _best(blackboard)
    rows = best.metadatos.get("analisis", {}).get("importancia_permutacion", [])
    return {
        "algorithm": best.algoritmo,
        "features": [
            {
                "feature": row["atributo"],
                "f1_drop_mean": round4(row["caida_f1_macro_media"]),
                "f1_drop_std": round4(row["caida_f1_macro_std"]),
            }
            for row in rows[:top]
        ],
    }


@tool("Matriz de confusión del mejor clasificador en prueba (filas: clase real; columnas: predicha).")
def confusion_matrix(blackboard) -> dict:
    best = _best(blackboard)
    matrix = best.matriz_confusion
    return {
        "algorithm": best.algoritmo,
        "labels": [str(label) for label in matrix.columns],
        "matrix": [[int(value) for value in row] for row in matrix.to_numpy()],
    }


CLASSIFICATION_TOOLS = (
    compare_classifiers,
    train_classifier,
    classification_feature_importance,
    confusion_matrix,
)
