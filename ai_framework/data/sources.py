"""Data loading and CSV source configuration module."""

from framework_ia.datos.fuentes import (
    CargadorCSV,
    ConfiguracionCSV,
)

CSVLoader = CargadorCSV
CSVConfig = ConfiguracionCSV

__all__ = ["CSVLoader", "CSVConfig", "CargadorCSV", "ConfiguracionCSV"]
