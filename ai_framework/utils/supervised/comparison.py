"""Comparison and dispatching of supervised models (classification and regression)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ai_framework.models.supervised import Supervisado


def dispatch_supervised_algorithm(
    modelo: Supervisado,
    algoritmo: str,
    **kwargs,
) -> Any:
    """Train and dispatch the requested supervised algorithm on the model instance."""
    return modelo.entrenar(algoritmo=algoritmo, **kwargs)


def compare_supervised_algorithms(
    modelo: Supervisado,
    algoritmos: list[str] | tuple[str, ...] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run comparison experiments on supervised models and return comparative summary."""
    experimento = modelo.experimentar(algoritmos=algoritmos, **kwargs)
    return experimento.tabla_comparativa()

# Backward compatibility aliases
despachar_algoritmo_supervisado = dispatch_supervised_algorithm
comparar_algoritmos_supervisados = compare_supervised_algorithms
