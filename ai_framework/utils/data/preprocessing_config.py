"""Configuración del preprocesamiento para algoritmos no supervisados."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Options to transform an arbitrary dataset into a numerical matrix."""

    columnas: tuple[str, ...] | None = None
    incluir_categoricas: bool = False
    imputar: bool = True
    estandarizar: bool = True

    @property
    def columns(self) -> tuple[str, ...] | None:
        return self.columnas

    @property
    def include_categorical(self) -> bool:
        return self.incluir_categoricas

    @property
    def impute(self) -> bool:
        return self.imputar

    @property
    def scale(self) -> bool:
        return self.estandarizar


ConfiguracionPreprocesamiento = PreprocessingConfig
__all__ = ['PreprocessingConfig', 'ConfiguracionPreprocesamiento']
