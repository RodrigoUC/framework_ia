"""Preparación local y trazable de Concrete Compressive Strength de UCI."""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

FUENTE_CONCRETE = (
    "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength"
)
ARCHIVO_ORIGINAL_CONCRETE = "Concrete_Data.xls"
ARCHIVO_CONCRETE = "concrete.csv"
FILAS_ORIGINALES_CONCRETE = 1030
PREDICTORAS_CONCRETE = (
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
)
OBJETIVO_CONCRETE = "strength"
COLUMNAS_CONCRETE = (*PREDICTORAS_CONCRETE, OBJETIVO_CONCRETE)
CABECERAS_ORIGINALES_CONCRETE = (
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)",
    "Concrete compressive strength(MPa, megapascals) ",
)
UNIDADES_CONCRETE = {
    **{columna: "kg/m^3" for columna in PREDICTORAS_CONCRETE[:-1]},
    "age": "day",
    "strength": "MPa",
}


def _sha256(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def _normalizar_cabecera(cabecera: object) -> str:
    """Tolera sólo diferencias de espacio y mayúsculas en nombres documentados."""
    if not isinstance(cabecera, str):
        raise TypeError("Concrete debe conservar las cabeceras originales de UCI.")
    return " ".join(cabecera.split()).casefold()


def preparar_concrete(contenido: bytes) -> tuple[dict[str, bytes], dict]:
    """Valida el XLS íntegro y elimina sólo duplicados de las nueve columnas."""
    if not contenido:
        raise ValueError("El archivo XLS original de Concrete está vacío.")
    originales = pd.read_excel(BytesIO(contenido), engine="xlrd", sheet_name=0)
    cabeceras = originales.columns.tolist()
    esperadas = [
        _normalizar_cabecera(nombre) for nombre in CABECERAS_ORIGINALES_CONCRETE
    ]
    recibidas = [_normalizar_cabecera(nombre) for nombre in cabeceras]
    if recibidas != esperadas:
        raise ValueError(
            "Concrete debe contener las nueve columnas originales de UCI "
            "en su orden original; revise cabeceras y archivo de origen."
        )
    if len(originales) != FILAS_ORIGINALES_CONCRETE:
        raise ValueError(
            f"El XLS original de Concrete debe tener {FILAS_ORIGINALES_CONCRETE} "
            f"filas; se recibieron {len(originales)}."
        )
    datos = originales.copy(deep=True)
    datos.columns = list(COLUMNAS_CONCRETE)
    if datos.isna().any().any():
        raise ValueError("Concrete no admite valores ausentes en el original.")
    for columna in COLUMNAS_CONCRETE:
        serie = datos[columna]
        if pd.api.types.is_bool_dtype(serie) or pd.api.types.is_complex_dtype(serie):
            raise ValueError(f"La columna {columna} debe contener números reales.")
        try:
            numerica = pd.to_numeric(serie, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"La columna {columna} debe ser numérica.") from exc
        if pd.api.types.is_bool_dtype(numerica) or pd.api.types.is_complex_dtype(
            numerica
        ):
            raise ValueError(f"La columna {columna} debe contener números reales.")
        datos[columna] = numerica.astype("float64")
    if not np.isfinite(datos.to_numpy()).all():
        raise ValueError("Concrete no admite valores infinitos ni ausentes.")
    duplicadas = datos.duplicated(keep="first")
    posiciones_eliminadas = np.flatnonzero(duplicadas.to_numpy()).tolist()
    preparado = datos.loc[~duplicadas].reset_index(drop=True)
    canonico = preparado.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata = {
        "dataset": "uci_concrete_compressive_strength",
        "archivo": ARCHIVO_CONCRETE,
        "fuente": FUENTE_CONCRETE,
        "licencia": {
            "nombre": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "referencia": FUENTE_CONCRETE,
        },
        "atribucion": {
            "titular": "Prof. I-Cheng Yeh",
            "publicacion": (
                "I-Cheng Yeh, Modeling of strength of high performance concrete using "
                "artificial neural networks, Cement and Concrete Research, "
                "28(12), 1797-1808 (1998)."
            ),
            "aviso_original": "Reuse with retention of copyright notice and published paper citation.",
        },
        "sha256": _sha256(canonico),
        "hash_sobre": "csv_materializado",
        "filas": len(preparado),
        "columnas": list(COLUMNAS_CONCRETE),
        "unidades": dict(UNIDADES_CONCRETE),
        "original": {
            "archivo": ARCHIVO_ORIGINAL_CONCRETE,
            "sha256": _sha256(contenido),
            "filas": len(originales),
            "columnas": cabeceras,
        },
        "mapeo_cabeceras": dict(zip(cabeceras, COLUMNAS_CONCRETE, strict=True)),
        "derivacion": {
            "script": "scripts/preparar_concrete.py",
            "criterio": "duplicados exactos sobre las nueve columnas; conservar primera fila",
            "filas_originales": len(datos),
            "filas_preparadas": len(preparado),
            "filas_eliminadas_duplicados": len(posiciones_eliminadas),
            "posiciones_originales_eliminadas": posiciones_eliminadas,
            "base_posiciones": 0,
            "incluye_cabecera_en_posiciones": False,
            "conversion_numerica": "float64; sin imputacion ni escalado",
        },
        "carga": {"funcion": "pandas.read_excel", "engine": "xlrd", "sheet_name": 0},
        "lectura": {
            "separador": ",",
            "decimal": ".",
            "encoding": "utf-8",
            "usar_primera_columna_como_indice": False,
        },
        "materializacion": {"indice": False, "fin_linea": "LF"},
        "preset_regresion": {
            "target": OBJETIVO_CONCRETE,
            "features": list(PREDICTORAS_CONCRETE),
            "feature_simple": "cement",
            "test_size": 0.25,
            "random_state": 42,
        },
        "versiones_origen": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xlrd": version("xlrd"),
        },
    }
    return {ARCHIVO_ORIGINAL_CONCRETE: contenido, ARCHIVO_CONCRETE: canonico}, metadata


def validar_preparacion_concrete(directorio: Path, metadata: dict) -> dict[str, bytes]:
    """Recalcula la derivación y exige igualdad de procedencia y de archivos."""
    directorio = Path(directorio)
    archivos, esperada = preparar_concrete(
        (directorio / ARCHIVO_ORIGINAL_CONCRETE).read_bytes()
    )
    if metadata != esperada:
        raise ValueError("La procedencia de Concrete no coincide con su derivación.")
    for nombre, contenido in archivos.items():
        if (directorio / nombre).read_bytes() != contenido:
            raise ValueError(f"Concrete: {nombre} no coincide con el original.")
    return archivos
