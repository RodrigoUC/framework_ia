"""Shared helpers for the regression and classification tools.

The supervised agents keep on the blackboard the domain model, the experiment
and the options used to run it, so the tools reuse exactly the same split and
preprocessing instead of guessing them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_ROWS = 20


def round4(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, 4)


def clean_name(name: str) -> str:
    """Drops the ColumnTransformer prefix: ``numericas__ph`` -> ``ph``."""
    return str(name).split("__", 1)[-1]


def training_options(blackboard) -> dict[str, Any]:
    """Split and preprocessing options the agent used, ready for ``entrenar``."""
    return dict(blackboard.results["supervised_options"])


def model_importance(model, columns) -> list[dict[str, Any]] | None:
    """Coefficients or impurity importances of a fitted estimator, if it exposes them."""
    importances = getattr(model, "feature_importances_", None)
    kind = "importance"
    if importances is None:
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            return None
        importances = np.atleast_2d(coefficients)
        importances = np.abs(importances).mean(axis=0) if importances.shape[0] > 1 else importances[0]
        kind = "coefficient"
    rows = [
        {"feature": clean_name(column), kind: round4(value)}
        for column, value in zip(columns, importances)
    ]
    return sorted(rows, key=lambda row: abs(row[kind] or 0), reverse=True)
