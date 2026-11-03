"""Comparison and dispatching of supervised models (classification and regression)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ai_framework.models.supervised import Supervisado


def despachar_algoritmo_supervisado(
    modelo: Supervisado,
    algoritmo: str,
    **kwargs,
) -> Any:
    """Train and dispatch the requested supervised algorithm on the model instance."""
    return modelo.entrenar(algoritmo=algoritmo, **kwargs)


def comparar_algoritmos_supervisados(
    modelo: Supervisado,
    algoritmos: list[str] | tuple[str, ...] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run comparison experiments on supervised models and return comparative summary."""
    experimento = modelo.experimentar(algoritmos=algoritmos, **kwargs)
    return experimento.tabla_comparativa()

# English aliases
dispatch_supervised_algorithm = despachar_algoritmo_supervisado
compare_supervised_algorithms = comparar_algoritmos_supervisados
