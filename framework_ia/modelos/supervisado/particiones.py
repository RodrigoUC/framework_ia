"""Re-exportación compatible de utilidades de partición desde ``datos.particion``."""

from __future__ import annotations

from ...datos.particion import (
    comprobar_clases,
    dividir_indices,
    huella_datos,
    validar_particion,
)

__all__ = [
    "comprobar_clases",
    "dividir_indices",
    "huella_datos",
    "validar_particion",
]
