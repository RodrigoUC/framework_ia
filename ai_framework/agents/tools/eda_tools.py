"""Exploratory analysis tools built on the methods inherited from :class:`EDA`.

They all return bounded summaries (no dataset rows) so the context sent to the
LLM stays small and does not expose individual records.
"""

from __future__ import annotations

import pandas as pd

from ai_framework.utils.agents.agent_config import EDAThresholds
from ai_framework.agents.tools.registry import tool

MAX_ROWS = 20


def task_type(series: pd.Series, thresholds: EDAThresholds = EDAThresholds()) -> str:
    """Classification if the target is categorical or numeric with few values.

    ``Supervised._tipo_target`` treats every numeric target as regression; here
    a 0/1 target (e.g. Potability) is recognized as classification.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if series.nunique(dropna=True) > thresholds.max_classification_classes:
            return "regresion"
    return "clasificacion"


def looks_like_identifier(series: pd.Series, min_rows: int) -> bool:
    """One distinct value per row, except for floats, which can be continuous."""
    rows = len(series)
    return (
        rows >= min_rows
        and series.nunique(dropna=True) == rows
        and not pd.api.types.is_float_dtype(series)
    )


def _round(value: float) -> float:
    return round(float(value), 4)


@tool(
    "Perfil del dataset: calidad, tipo de cada columna, nulos, cardinalidad y "
    "relación de las predictoras con el target."
)
def dataset_profile(blackboard) -> dict:
    profile: dict = {
        "quality": blackboard.resumen_calidad(),
        "columns": blackboard.resumen_columnas().reset_index().to_dict("records"),
    }
    if blackboard.target is not None:
        series = blackboard.datos[blackboard.target]
        target: dict = {"name": blackboard.target, "task": task_type(series)}
        if target["task"] == "clasificacion":
            summary, _ = blackboard.balance_objetivo(blackboard.target)
            target["classes"] = {
                str(label): _round(share) for label, share in summary["Proporción"].items()
            }
        if pd.api.types.is_numeric_dtype(series):
            numeric = blackboard.datos.select_dtypes(include="number").drop(
                columns=[blackboard.target]
            )
            correlations = numeric.corrwith(series).dropna()
            target["correlations"] = {
                str(c): _round(v)
                for c, v in correlations.sort_values(key=abs, ascending=False).items()
            }
        profile["target"] = target
    blackboard.results["profile"] = profile
    return profile


@tool(
    "Pares de variables numéricas cuya correlación absoluta supera el umbral.",
    threshold="Correlación absoluta mínima, entre 0 y 1.",
    method="pearson, spearman o kendall.",
)
def correlations(blackboard, threshold: float = 0.8, method: str = "pearson") -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")
    matrix = blackboard.datos.select_dtypes(include="number").corr(method=method)
    pairs = []
    columns = list(matrix.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = matrix.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"a": str(a), "b": str(b), "r": _round(value)})
    pairs.sort(key=lambda pair: abs(pair["r"]), reverse=True)
    return {"method": method, "threshold": threshold, "pairs": pairs[:MAX_ROWS], "total": len(pairs)}


@tool(
    "Valores atípicos por rango intercuartílico en columnas numéricas.",
    columns="Columnas a revisar; si se omite, todas las numéricas.",
    iqr_factor="Multiplicador del IQR para los límites (1.5 es el estándar).",
)
def outliers(blackboard, columns: list[str] | None = None, iqr_factor: float = 1.5) -> dict:
    summary = blackboard.detectar_outliers(columns, iqr_factor)["resumen"]
    summary = summary.sort_values("porcentaje_outliers", ascending=False).head(MAX_ROWS)
    return {
        "iqr_factor": iqr_factor,
        "columns": [
            {
                "column": str(column),
                "count": int(row["cantidad_outliers"]),
                "percentage": float(row["porcentaje_outliers"]),
                "lower_limit": _round(row["limite_inferior"]),
                "upper_limit": _round(row["limite_superior"]),
            }
            for column, row in summary.iterrows()
        ],
    }


@tool(
    "Frecuencias de los valores más comunes de una columna.",
    column="Nombre exacto de la columna.",
    top="Cantidad máxima de valores a devolver.",
)
def column_frequencies(blackboard, column: str, top: int = 10) -> dict:
    if column not in blackboard.datos.columns:
        raise KeyError(f"No existe la columna {column!r}.")
    table = blackboard.frecuencias([column])[column].head(top)
    return {
        "column": column,
        "unique_values": int(blackboard.datos[column].nunique(dropna=False)),
        "frequencies": [
            {"value": str(value), "frequency": int(f["frecuencia"]), "percentage": float(f["porcentaje"])}
            for value, f in table.iterrows()
        ],
    }


EDA_TOOLS = (dataset_profile, correlations, outliers, column_frequencies)
