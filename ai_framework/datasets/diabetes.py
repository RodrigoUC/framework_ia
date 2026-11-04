"""Preparación y referencia del dataset Diabetes para clasificación y regresión."""

from __future__ import annotations

import csv
import hashlib
import platform
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_diabetes

FUENTE_DIABETES = "https://hbiostat.org/data/repo/diabetes"
PREDICTORAS_DIABETES = (
    "chol",
    "stab.glu",
    "hdl",
    "ratio",
    "location",
    "age",
    "gender",
    "height",
    "weight",
    "frame",
    "bp.1s",
    "bp.1d",
    "bp.2s",
    "bp.2d",
    "waist",
    "hip",
    "time.ppn",
)
CATEGORICAS_DIABETES = ("location", "gender", "frame")
COLUMNAS_ORIGINALES = (
    "id",
    "chol",
    "stab.glu",
    "hdl",
    "ratio",
    "glyhb",
    "location",
    "age",
    "gender",
    "height",
    "weight",
    "frame",
    "bp.1s",
    "bp.1d",
    "bp.2s",
    "bp.2d",
    "waist",
    "hip",
    "time.ppn",
)

FUENTE_DIABETES_REGRESION = (
    "https://scikit-learn.org/stable/modules/generated/"
    "sklearn.datasets.load_diabetes.html"
)
ARCHIVO_DIABETES_REGRESION = "diabetes_regresion.csv"


def _sha(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def preparar_diabetes(contenido: bytes) -> tuple[dict[str, bytes], dict]:
    """Conserva el original y deriva clases sin ajustar estadísticas de predictoras."""
    try:
        for fila in csv.reader(StringIO(contenido.decode("utf-8-sig")), strict=True):
            if fila and len(fila) != len(COLUMNAS_ORIGINALES):
                raise ValueError(
                    "El original debe contener las 19 columnas de Diabetes sin campos extra."
                )
    except csv.Error as exc:
        raise ValueError("El original no tiene una estructura CSV válida.") from exc
    datos = pd.read_csv(BytesIO(contenido), sep=",", decimal=".", encoding="utf-8")
    if (
        datos.empty
        or not isinstance(datos.index, pd.RangeIndex)
        or len(datos.columns) != len(COLUMNAS_ORIGINALES)
        or set(datos.columns) != set(COLUMNAS_ORIGINALES)
    ):
        raise ValueError(
            "El original debe contener las 19 columnas de Diabetes sin campos extra."
        )
    ids = pd.to_numeric(datos["id"], errors="raise")
    if (
        ids.isna().any()
        or not np.isfinite(ids.to_numpy()).all()
        or not ids.eq(ids.round()).all()
        or not ids.is_unique
    ):
        raise ValueError("Los ID originales deben ser enteros únicos no faltantes.")
    glyhb = pd.to_numeric(datos["glyhb"], errors="raise")
    if not np.isfinite(glyhb.dropna().to_numpy()).all():
        raise ValueError("glyhb contiene valores no finitos.")
    for columna in PREDICTORAS_DIABETES:
        if columna not in CATEGORICAS_DIABETES:
            pd.to_numeric(datos[columna], errors="raise")
    conocidos = glyhb.notna()
    if not conocidos.any():
        raise ValueError("No hay filas con glyhb observado para construir etiquetas.")
    preparado = datos.loc[conocidos, list(PREDICTORAS_DIABETES)].copy()
    preparado["diabetes"] = np.where(
        glyhb.loc[conocidos].gt(7.0), "Diabetes", "No diabetes"
    )
    preparado = preparado.reset_index(drop=True)
    trazabilidad = pd.DataFrame(
        {
            "fila_preparada": np.arange(len(preparado)),
            "fila_original": datos.index[conocidos].to_numpy(),
            "id_original": ids.loc[conocidos].astype("int64").to_numpy(),
        }
    )
    archivos = {
        "diabetes_original.csv": contenido,
        "diabetes.csv": preparado.to_csv(index=False, lineterminator="\n").encode(
            "utf-8"
        ),
        "trazabilidad_diabetes.csv": trazabilidad.to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8"),
    }
    metadata = {
        "ruta": "diabetes.csv",
        "fuente": FUENTE_DIABETES,
        "sha256": _sha(archivos["diabetes.csv"]),
        "original": {
            "ruta": "diabetes_original.csv",
            "sha256": _sha(contenido),
            "filas": len(datos),
            "columnas": list(datos.columns),
        },
        "derivacion": {
            "script": "scripts/preparar_diabetes.py",
            "regla": {
                "columna": "glyhb",
                "operador": ">",
                "umbral": 7.0,
                "positiva": "Diabetes",
                "negativa": "No diabetes",
            },
            "filas_originales": len(datos),
            "filas_preparadas": len(preparado),
            "excluidos_sin_glyhb": [
                {"fila_original": int(fila), "id_original": int(ids.loc[fila])}
                for fila in datos.index[~conocidos]
            ],
            "predictoras": list(PREDICTORAS_DIABETES),
            "categoricas": list(CATEGORICAS_DIABETES),
            "clases": {
                etiqueta: int(preparado.diabetes.eq(etiqueta).sum())
                for etiqueta in ("Diabetes", "No diabetes")
            },
            "trazabilidad": {
                "ruta": "trazabilidad_diabetes.csv",
                "sha256": _sha(archivos["trazabilidad_diabetes.csv"]),
                "filas": len(trazabilidad),
            },
            "lectura": {"separador": ",", "decimal": ".", "encoding": "utf-8"},
        },
    }
    return archivos, metadata


def validar_preparacion_diabetes(directorio: Path, metadata: dict) -> dict[str, bytes]:
    """Comprueba original, derivación y mapa regenerándolos desde los mismos bytes."""
    directorio = Path(directorio)
    if not isinstance(metadata, dict):
        raise TypeError("La procedencia de Diabetes debe ser un objeto.")
    archivos, esperada = preparar_diabetes(
        (directorio / "diabetes_original.csv").read_bytes()
    )
    if metadata != esperada:
        raise ValueError("La procedencia de Diabetes no coincide con su derivación.")
    for nombre, contenido in archivos.items():
        if (directorio / nombre).read_bytes() != contenido:
            raise ValueError(
                f"Diabetes: el archivo {nombre} no coincide con el original y la regla."
            )
    return archivos


def cargar_diabetes_regresion() -> tuple[pd.DataFrame, bytes, dict]:
    """Devuelve datos crudos, CSV canónico y procedencia; no ajusta transformaciones."""
    referencia = load_diabetes(as_frame=True, scaled=False)
    datos = referencia.frame.copy(deep=True)
    contenido = datos.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata = {
        "dataset": "sklearn_diabetes_regression",
        "archivo": ARCHIVO_DIABETES_REGRESION,
        "fuente": FUENTE_DIABETES_REGRESION,
        "sha256": _sha(contenido),
        "hash_sobre": "csv_materializado",
        "filas": len(datos),
        "columnas": datos.columns.tolist(),
        "carga": {"funcion": "load_diabetes", "as_frame": True, "scaled": False},
        "lectura": {
            "separador": ",",
            "decimal": ".",
            "encoding": "utf-8",
            "usar_primera_columna_como_indice": False,
        },
        "materializacion": {"indice": False, "fin_linea": "LF"},
        "preset_regresion": {
            "target": "target",
            "features": list(referencia.feature_names),
            "feature_simple": "bmi",
            "test_size": 0.25,
            "random_state": 42,
        },
        "versiones_origen": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    return datos, contenido, metadata
