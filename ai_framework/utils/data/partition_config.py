"""Configuration for splitting a dataset into train/test/validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionConfig:
    """Percentages and options to split a dataset into train/test/validation subsets."""

    porcentaje_test: float = 0.2
    porcentaje_validacion: float = 0.0
    columna_estratificacion: str | None = None
    semilla: int = 42

    @property
    def test_size(self) -> float:
        return self.porcentaje_test

    @property
    def validation_size(self) -> float:
        return self.porcentaje_validacion

    @property
    def stratify_column(self) -> str | None:
        return self.columna_estratificacion

    @property
    def random_state(self) -> int:
        return self.semilla


ConfiguracionParticion = PartitionConfig
__all__ = ['PartitionConfig', 'ConfiguracionParticion']
