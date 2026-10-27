"""Preparación reproducible del dataset Potabilidad (Water Quality)."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

FUENTE_POTABILIDAD = "https://www.kaggle.com/datasets/adityakadiwal/water-potability"
PREDICTORAS_POTABILIDAD = (
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
)
OBJETIVO_POTABILIDAD = "Potability"


def _sha(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def preparar_potabilidad(contenido: bytes) -> tuple[dict[str, bytes], dict]:
    """Devuelve el original, los casos completos y su procedencia."""
    datos = pd.read_csv(BytesIO(contenido), sep=",", decimal=".", encoding="utf-8")
    esperadas = [*PREDICTORAS_POTABILIDAD, OBJETIVO_POTABILIDAD]
    if list(datos.columns) != esperadas:
        raise ValueError(f"Potabilidad debe tener las columnas {esperadas}.")
    if not datos[OBJETIVO_POTABILIDAD].dropna().isin([0, 1]).all():
        raise ValueError("Potability debe contener sólo 0 y 1.")
    completos = datos.dropna().reset_index(drop=True)
    completos[OBJETIVO_POTABILIDAD] = completos[OBJETIVO_POTABILIDAD].astype(int)
    preparado = completos.to_csv(index=False, lineterminator="\n").encode("utf-8")
    archivos = {"water_potability.csv": contenido, "potabilidad.csv": preparado}
    metadata = {
        "ruta": "potabilidad.csv",
        "fuente": FUENTE_POTABILIDAD,
        "sha256": _sha(preparado),
        "original": {
            "ruta": "water_potability.csv",
            "sha256": _sha(contenido),
            "filas": len(datos),
            "ausentes_por_columna": {
                columna: int(valor)
                for columna, valor in datos.isna().sum().items()
                if valor
            },
        },
        "derivacion": {
            "script": "scripts/preparar_potabilidad.py",
            "criterio": "casos completos (filas sin valores ausentes)",
            "filas_originales": len(datos),
            "filas_preparadas": len(completos),
            "clases": {
                str(clase): int(cantidad)
                for clase, cantidad in completos[OBJETIVO_POTABILIDAD]
                .value_counts()
                .sort_index()
                .items()
            },
            "significado_clases": {"0": "No potable", "1": "Potable"},
        },
    }
    return archivos, metadata


def validar_preparacion_potabilidad(directorio: Path, metadata: dict) -> dict[str, bytes]:
    """Regenera la preparación desde el original y exige igualdad de bytes."""
    directorio = Path(directorio)
    archivos, esperada = preparar_potabilidad(
        (directorio / "water_potability.csv").read_bytes()
    )
    if metadata != esperada:
        raise ValueError("La procedencia de Potabilidad no coincide con su derivación.")
    for nombre, contenido in archivos.items():
        if (directorio / nombre).read_bytes() != contenido:
            raise ValueError(f"Potabilidad: {nombre} no coincide con el original.")
    return archivos
