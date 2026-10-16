"""Preparación de Potabilidad: casos completos reproducibles, sin imputación."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from framework_ia.datos.potabilidad_lab02 import (
    PREDICTORAS_POTABILIDAD,
    preparar_potabilidad,
    validar_preparacion_potabilidad,
)
from scripts.preparar_potabilidad_lab02 import ejecutar


def original_fixture():
    datos = pd.DataFrame(
        {columna: np.arange(6, dtype=float) + i for i, columna in enumerate(PREDICTORAS_POTABILIDAD)}
    )
    datos.loc[0, "ph"] = np.nan
    datos.loc[3, "Sulfate"] = np.nan
    datos["Potability"] = [0, 1, 0, 1, 0, 1]
    return datos.to_csv(index=False, lineterminator="\n").encode("utf-8")


def test_conserva_original_y_solo_elimina_filas_incompletas():
    archivos, metadata = preparar_potabilidad(original_fixture())
    preparado = pd.read_csv(pd.io.common.BytesIO(archivos["potabilidad.csv"]))
    assert archivos["water_potability.csv"] == original_fixture()
    assert len(preparado) == 4
    assert not preparado.isna().any().any()
    assert metadata["derivacion"]["filas_originales"] == 6
    assert metadata["derivacion"]["filas_preparadas"] == 4
    assert metadata["original"]["ausentes_por_columna"] == {"ph": 1, "Sulfate": 1}
    assert preparar_potabilidad(original_fixture()) == (archivos, metadata)


def test_rechaza_columnas_o_clases_inesperadas():
    datos = pd.read_csv(pd.io.common.BytesIO(original_fixture()))
    with pytest.raises(ValueError):
        preparar_potabilidad(datos.drop(columns="ph").to_csv(index=False).encode())
    datos["Potability"] = [0, 1, 2, 1, 0, 1]
    with pytest.raises(ValueError):
        preparar_potabilidad(datos.to_csv(index=False).encode())


def test_cli_preserva_diabetes_en_procedencia_y_valida(tmp_path):
    origen = tmp_path / "agua.csv"
    origen.write_bytes(original_fixture())
    destino = tmp_path / "datos"
    destino.mkdir()
    (destino / "procedencia.json").write_text(json.dumps({"diabetes": {"x": 1}}), encoding="utf-8")
    ejecutar(argparse.Namespace(csv=origen, salida=destino))
    procedencia = json.loads((destino / "procedencia.json").read_text(encoding="utf-8"))
    assert procedencia["diabetes"] == {"x": 1}
    validar_preparacion_potabilidad(destino, procedencia["potabilidad"])
    (destino / "potabilidad.csv").write_bytes(b"alterado\n")
    with pytest.raises(ValueError, match="no coincide"):
        validar_preparacion_potabilidad(destino, procedencia["potabilidad"])
