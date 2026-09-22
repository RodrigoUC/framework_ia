"""Compatibilidad: use ``framework_ia.modelos.no_supervisado``."""

from .no_supervisado.reduccion_dimensional import (
    DependenciaOpcionalError,
    ReduccionDimensional,
)

__all__ = ["DependenciaOpcionalError", "ReduccionDimensional"]
