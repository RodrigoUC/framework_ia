"""Data partitioning and splitting module."""

from framework_ia.datos.particion import (
    ConfiguracionParticion,
    Particionador,
)

DataSplitter = Particionador
PartitionConfig = ConfiguracionParticion

__all__ = ["DataSplitter", "PartitionConfig", "Particionador", "ConfiguracionParticion"]
