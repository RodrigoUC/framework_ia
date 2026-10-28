"""Comparación y despacho de modelos supervisados (clasificación y regresión)."""

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
    """Entrena y despacha el algoritmo supervisado solicitado sobre la instancia."""
    return modelo.entrenar(algoritmo=algoritmo, **kwargs)


def comparar_algoritmos_supervisados(
    modelo: Supervisado,
    algoritmos: list[str] | tuple[str, ...] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Ejecuta experimentos de comparación sobre modelos supervisados y retorna resumen comparativo."""
    experimento = modelo.experimentar(algoritmos=algoritmos, **kwargs)
    return experimento.tabla_comparativa()
