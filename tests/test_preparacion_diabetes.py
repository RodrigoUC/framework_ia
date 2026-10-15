"""Casos de frontera, trazabilidad y conservación del original de Diabetes."""

import argparse
import hashlib
import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from framework_ia.datos.diabetes_lab02 import (
    COLUMNAS_ORIGINALES,
    PREDICTORAS_DIABETES,
    preparar_diabetes,
    validar_preparacion_diabetes,
)
from scripts.preparar_diabetes_lab02 import ejecutar


def original_fixture():
    datos = pd.DataFrame({columna: [1, 2, 3, 4] for columna in COLUMNAS_ORIGINALES})
    datos["id"] = [100, 200, 300, 400]
    datos["glyhb"] = [6.99, 7.0, 7.01, np.nan]
    datos["gender"] = ["female", "male", "female", "male"]
    datos["location"] = ["Buckingham"] * 4
    datos["frame"] = ["small", "large", None, "medium"]
    datos["bp.2s"] = [np.nan] * 4
    return datos.to_csv(index=False, lineterminator="\n").encode("utf-8")


def test_umbral_estricto_no_imputa_etiquetas_y_preserva_faltantes():
    original = original_fixture()
    archivos, metadata = preparar_diabetes(original)
    datos = pd.read_csv(BytesIO(archivos["diabetes.csv"]))
    assert archivos["diabetes_original.csv"] == original
    assert list(datos.columns) == [*PREDICTORAS_DIABETES, "diabetes"]
    assert datos.diabetes.tolist() == ["No diabetes", "No diabetes", "Diabetes"]
    assert datos["bp.2s"].isna().all()
    assert pd.isna(datos.frame.iloc[2])
    assert datos.chol.tolist() == [1, 2, 3]
    assert metadata["original"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["derivacion"]["excluidos_sin_glyhb"] == [
        {"fila_original": 3, "id_original": 400}
    ]
    assert preparar_diabetes(original) == (archivos, metadata)


def test_mapa_relaciona_indices_con_ids_originales():
    datos = pd.read_csv(BytesIO(original_fixture()))
    datos.loc[1, "glyhb"] = np.nan
    datos.loc[3, "glyhb"] = 8.0
    archivos, _ = preparar_diabetes(datos.to_csv(index=False).encode("utf-8"))
    mapa = pd.read_csv(BytesIO(archivos["trazabilidad_diabetes.csv"]))
    assert mapa.to_dict("list") == {
        "fila_preparada": [0, 1, 2],
        "fila_original": [0, 2, 3],
        "id_original": [100, 300, 400],
    }


def test_rechaza_campos_extra_que_pandas_interpreta_como_indice():
    lineas = original_fixture().decode("utf-8").splitlines()
    original = "\n".join(
        [lineas[0], *(f"{99 + fila},{linea}" for fila, linea in enumerate(lineas[1:]))]
    ).encode("utf-8")
    with pytest.raises(ValueError, match="sin campos extra"):
        preparar_diabetes(original)


@pytest.mark.parametrize("cambio", ["id_duplicado", "glyhb_texto", "glyhb_infinito", "sin_objetivo", "columna_ausente"])
def test_original_invalido_no_produce_etiquetas(cambio):
    datos = pd.read_csv(BytesIO(original_fixture()))
    if cambio == "id_duplicado":
        datos.loc[1, "id"] = datos.loc[0, "id"]
    elif cambio == "glyhb_texto":
        datos["glyhb"] = ["incorrecto", "7", "8", None]
    elif cambio == "glyhb_infinito":
        datos.loc[0, "glyhb"] = np.inf
    elif cambio == "sin_objetivo":
        datos["glyhb"] = np.nan
    else:
        datos = datos.drop(columns="chol")
    with pytest.raises((ValueError, TypeError)):
        preparar_diabetes(datos.to_csv(index=False).encode("utf-8"))


def test_cli_idempotente_preserva_potabilidad_y_valida_toda_la_derivacion(tmp_path):
    origen = tmp_path / "entrada.csv"
    origen.write_bytes(original_fixture())
    destino = tmp_path / "datos"
    destino.mkdir()
    agua = {"ruta": "agua.csv", "sha256": "agua-sin-cambios"}
    (destino / "procedencia.json").write_text(
        json.dumps({"potabilidad": agua}), encoding="utf-8"
    )
    argumentos = argparse.Namespace(csv=origen, salida=destino)
    ejecutar(argumentos)
    antes = {ruta.name: ruta.read_bytes() for ruta in destino.iterdir()}
    ejecutar(argumentos)
    assert antes == {ruta.name: ruta.read_bytes() for ruta in destino.iterdir()}
    metadata = json.loads((destino / "procedencia.json").read_text(encoding="utf-8"))
    assert metadata["potabilidad"] == agua
    validar_preparacion_diabetes(destino, metadata["diabetes"])
    (destino / "trazabilidad_diabetes.csv").write_bytes(b"mapa incorrecto\n")
    with pytest.raises(ValueError, match="trazabilidad"):
        validar_preparacion_diabetes(destino, metadata["diabetes"])


def test_cli_rechaza_sobrescritura_sin_escrituras_parciales(tmp_path):
    origen = tmp_path / "entrada.csv"
    origen.write_bytes(original_fixture())
    destino = tmp_path / "datos"
    destino.mkdir()
    existente = destino / "diabetes.csv"
    existente.write_bytes(b"evidencia previa\n")
    with pytest.raises(ValueError, match="archivo distinto"):
        ejecutar(argparse.Namespace(csv=origen, salida=destino))
    assert existente.read_bytes() == b"evidencia previa\n"
    assert sorted(r.name for r in destino.iterdir()) == ["diabetes.csv"]


def test_validador_rechaza_cambio_de_regla_en_procedencia(tmp_path):
    archivos, metadata = preparar_diabetes(original_fixture())
    for nombre, contenido in archivos.items():
        (tmp_path / nombre).write_bytes(contenido)
    metadata["derivacion"]["regla"]["umbral"] = 6.5
    with pytest.raises(ValueError, match="procedencia"):
        validar_preparacion_diabetes(tmp_path, metadata)
