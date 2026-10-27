"""Exportación unificada de evidencia de modelos y experimentos supervisados."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .exportacion_lab02 import (
    _archivos_analisis,
    _bytes_json,
    _json_seguro,
    _versiones,
    construir_archivos_experimento as construir_archivos_experimento_clasificacion,
    construir_zip_experimento,
)
from .exportacion_lab03 import (
    construir_archivos_experimento as construir_archivos_experimento_regresion,
    construir_archivos_modelo,
)
from .resultados import (
    ResultadoExperimentoClasificacion,
    ResultadoExperimentoRegresion,
)

__all__ = [
    "_archivos_analisis",
    "_bytes_json",
    "_json_seguro",
    "_versiones",
    "construir_archivos_experimento",
    "construir_archivos_experimento_clasificacion",
    "construir_archivos_experimento_regresion",
    "construir_archivos_modelo",
    "construir_zip_experimento",
]


def construir_archivos_experimento(
    resultado: ResultadoExperimentoClasificacion | ResultadoExperimentoRegresion,
    *,
    argumentos: Mapping[str, Any],
    dataset: Mapping[str, Any],
    versiones: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Serializa un experimento supervisado (clasificación o regresión) según su tipo."""
    if isinstance(resultado, ResultadoExperimentoRegresion) or (
        not isinstance(resultado, ResultadoExperimentoClasificacion)
        and hasattr(resultado, "mejores")
        and any(hasattr(v, "metricas_train") for v in resultado.mejores.values())
    ):
        return construir_archivos_experimento_regresion(
            resultado, argumentos=argumentos, dataset=dataset, versiones=versiones
        )
    return construir_archivos_experimento_clasificacion(
        resultado, argumentos=argumentos, dataset=dataset, versiones=versiones
    )
