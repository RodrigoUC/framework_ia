"""Re-exportación compatible del módulo unificado de evidencia."""

from __future__ import annotations

from .evidencia import (
    METRICAS,
    EvidenciaLab03,
    EvidenciaRegresion,
    _cargar,
    _cerca,
    _comparacion,
    _cv_interna,
    _exigir,
    _indices,
    _json,
    _normalizar,
    _objeto,
    _particiones,
    _protocolo,
    _validar_dataset,
    cargar_evidencia,
)

__all__ = [
    "EvidenciaLab03",
    "EvidenciaRegresion",
    "METRICAS",
    "cargar_evidencia",
]
