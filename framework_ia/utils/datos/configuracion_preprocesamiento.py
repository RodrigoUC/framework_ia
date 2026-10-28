"""Configuración del preprocesamiento para algoritmos no supervisados."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionPreprocesamiento:
    """Opciones para convertir un dataset arbitrario en una matriz numérica."""

    columnas: tuple[str, ...] | None = None
    incluir_categoricas: bool = False
    imputar: bool = True
    estandarizar: bool = True
