"""Configuration for reading CSV files."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CSVConfig:
    """Explicit parameters to parse a CSV file."""

    separador: str = ","
    decimal: str = "."
    encoding: str = "utf-8"
    usar_primera_columna_como_indice: bool = False

    @property
    def separator(self) -> str:
        return self.separador

    @property
    def use_first_column_as_index(self) -> bool:
        return self.usar_primera_columna_como_indice


ConfiguracionCSV = CSVConfig
__all__ = ['CSVConfig', 'ConfiguracionCSV']
