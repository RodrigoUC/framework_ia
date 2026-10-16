"""Referencia de regresión incluida en scikit-learn, sin acceso a la red."""

from __future__ import annotations

import hashlib
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_diabetes

FUENTE_DIABETES_REGRESION = (
    "https://scikit-learn.org/stable/modules/generated/"
    "sklearn.datasets.load_diabetes.html"
)
ARCHIVO_DIABETES_REGRESION = "diabetes_regresion.csv"


def cargar_diabetes_regresion() -> tuple[pd.DataFrame, bytes, dict]:
    """Devuelve datos crudos, CSV canónico y procedencia; no ajusta transformaciones."""
    referencia = load_diabetes(as_frame=True, scaled=False)
    datos = referencia.frame.copy(deep=True)
    contenido = datos.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata = {
        "dataset": "sklearn_diabetes_regression",
        "archivo": ARCHIVO_DIABETES_REGRESION,
        "fuente": FUENTE_DIABETES_REGRESION,
        "sha256": hashlib.sha256(contenido).hexdigest(),
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
