"""Configuración para dividir un dataset en train/test/validación."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionParticion:
    """Porcentajes y opciones para dividir un dataset en subconjuntos."""

    porcentaje_test: float = 0.2
    porcentaje_validacion: float = 0.0
    columna_estratificacion: str | None = None
    semilla: int = 42
