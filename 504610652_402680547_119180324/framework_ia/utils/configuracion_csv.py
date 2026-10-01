"""Configuración para la lectura de archivos CSV."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionCSV:
    """Parámetros explícitos para interpretar un archivo CSV."""

    separador: str = ","
    decimal: str = "."
    encoding: str = "utf-8"
    usar_primera_columna_como_indice: bool = False
