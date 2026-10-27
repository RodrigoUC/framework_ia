"""Carga configurable de archivos CSV desde rutas o contenido en memoria."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from ..utils import ConfiguracionCSV
from .dataframe import DataFrame


class CargadorCSV:
    """Adapta distintas fuentes CSV a un pandas.DataFrame o DataFrame del framework."""

    @staticmethod
    def cargar_ruta(ruta: str | Path, configuracion: ConfiguracionCSV) -> pd.DataFrame:
        """Carga un CSV disponible en el sistema de archivos."""
        ruta = Path(ruta)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo '{ruta}'.")
        return CargadorCSV._leer(ruta, configuracion)

    @staticmethod
    def cargar_bytes(contenido: bytes, configuracion: ConfiguracionCSV) -> pd.DataFrame:
        """Carga un CSV subido por el usuario sin escribirlo en disco."""
        if not contenido:
            raise ValueError("El archivo CSV está vacío.")
        return CargadorCSV._leer(BytesIO(contenido), configuracion)

    @staticmethod
    def cargar_dataframe(ruta: str | Path, configuracion: ConfiguracionCSV) -> DataFrame:
        """Carga un CSV desde ruta y lo envuelve en la clase ``DataFrame`` del framework."""
        return DataFrame(dataframe=CargadorCSV.cargar_ruta(ruta, configuracion))

    @staticmethod
    def cargar_bytes_dataframe(
        contenido: bytes, configuracion: ConfiguracionCSV
    ) -> DataFrame:
        """Carga un CSV desde bytes y lo envuelve en la clase ``DataFrame`` del framework."""
        return DataFrame(dataframe=CargadorCSV.cargar_bytes(contenido, configuracion))

    @staticmethod
    def _leer(origen, configuracion: ConfiguracionCSV) -> pd.DataFrame:
        """Centraliza la llamada a pandas para mantener igual comportamiento."""
        indice = 0 if configuracion.usar_primera_columna_como_indice else None
        datos = pd.read_csv(
            origen,
            sep=configuracion.separador,
            decimal=configuracion.decimal,
            encoding=configuracion.encoding,
            index_col=indice,
        )
        if datos.empty:
            raise ValueError("El CSV no contiene registros analizables.")
        return datos
