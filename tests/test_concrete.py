"""Preparación de Concrete: datos intactos, derivación exacta y escritura segura."""

import argparse
import hashlib
import json
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from framework_ia.datos import concrete
from scripts.preparar_concrete import crear_parser, ejecutar, main


@pytest.fixture
def originales():
    datos = pd.DataFrame(
        {
            columna: np.arange(1030, dtype=float) / 10 + posicion
            for posicion, columna in enumerate(concrete.CABECERAS_ORIGINALES_CONCRETE)
        }
    )
    datos.iloc[9] = datos.iloc[1]
    datos.iloc[44] = datos.iloc[1]
    datos.iloc[8, :-1] = datos.iloc[2, :-1]
    datos.iloc[8, -1] = datos.iloc[2, -1] + 100
    return datos


@pytest.fixture
def lectura_xls(monkeypatch, originales):
    def leer(origen, *, engine, sheet_name):
        assert isinstance(origen, BytesIO)
        assert origen.getvalue() == b"original-xls-bytes"
        assert (engine, sheet_name) == ("xlrd", 0)
        return originales.copy(deep=True)

    monkeypatch.setattr(concrete.pd, "read_excel", leer)
    monkeypatch.setattr(concrete, "version", lambda nombre: "fixture-xlrd")
    return originales


def test_prepara_sin_red_preserva_original_y_conserva_distinta_respuesta(lectura_xls):
    antes = lectura_xls.copy(deep=True)
    with patch("socket.create_connection", side_effect=AssertionError("network")):
        archivos, metadata = concrete.preparar_concrete(b"original-xls-bytes")
        assert concrete.preparar_concrete(b"original-xls-bytes") == (archivos, metadata)
    pd.testing.assert_frame_equal(lectura_xls, antes)
    assert archivos[concrete.ARCHIVO_ORIGINAL_CONCRETE] == b"original-xls-bytes"
    contenido = archivos[concrete.ARCHIVO_CONCRETE]
    preparado = pd.read_csv(BytesIO(contenido))
    esperado = antes.drop_duplicates(keep="first").reset_index(drop=True)
    esperado.columns = list(concrete.COLUMNAS_CONCRETE)
    pd.testing.assert_frame_equal(preparado, esperado)
    assert len(preparado) == 1028
    assert (
        preparado.loc[2, list(concrete.PREDICTORAS_CONCRETE)]
        == preparado.loc[8, list(concrete.PREDICTORAS_CONCRETE)]
    ).all()
    assert preparado.loc[2, "strength"] != preparado.loc[8, "strength"]
    assert b"\r" not in contenido and contenido.endswith(b"\n")
    assert metadata["sha256"] == hashlib.sha256(contenido).hexdigest()
    assert metadata["hash_sobre"] == "csv_materializado"
    assert (
        metadata["original"]["sha256"]
        == hashlib.sha256(b"original-xls-bytes").hexdigest()
    )
    assert metadata["original"]["archivo"] == "Concrete_Data.xls"
    assert metadata["derivacion"]["posiciones_originales_eliminadas"] == [9, 44]
    assert metadata["derivacion"]["filas_eliminadas_duplicados"] == 2
    assert metadata["derivacion"]["base_posiciones"] == 0
    assert metadata["preset_regresion"]["feature_simple"] == "cement"
    assert metadata["preset_regresion"]["features"] == list(
        concrete.PREDICTORAS_CONCRETE
    )
    assert metadata["unidades"]["strength"] == "MPa"
    assert metadata["unidades"]["age"] == "day"
    assert metadata["unidades"]["cement"] == "kg/m^3"
    assert metadata["mapeo_cabeceras"][antes.columns[-1]] == "strength"
    assert metadata["licencia"]["referencia"] == concrete.FUENTE_CONCRETE
    assert metadata["atribucion"]["titular"] == "Prof. I-Cheng Yeh"
    assert set(metadata["versiones_origen"]) == {"python", "numpy", "pandas", "xlrd"}
    serializado = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    assert "timestamp" not in serializado and "C:\\" not in serializado


@pytest.mark.parametrize("cambio", ["orden", "nombre", "faltante", "extra", "filas"])
def test_rechaza_otro_esquema_o_cantidad_original(monkeypatch, originales, cambio):
    if cambio == "orden":
        originales = originales.iloc[:, ::-1]
    elif cambio == "nombre":
        originales = originales.rename(columns={originales.columns[0]: "cement"})
    elif cambio == "faltante":
        originales = originales.iloc[:, :-1]
    elif cambio == "extra":
        originales = originales.assign(extra=1)
    else:
        originales = originales.iloc[:-1]
    monkeypatch.setattr(concrete.pd, "read_excel", lambda *args, **kwargs: originales)
    with pytest.raises(ValueError, match="columnas originales|1030"):
        concrete.preparar_concrete(b"source")


@pytest.mark.parametrize("cambio", ["nan", "infinito", "texto", "booleano", "complejo"])
def test_rechaza_valores_no_reales_finitos(monkeypatch, originales, cambio):
    columna = originales.columns[0]
    if cambio == "nan":
        originales.loc[0, columna] = np.nan
    elif cambio == "infinito":
        originales.loc[0, columna] = np.inf
    elif cambio == "texto":
        originales[columna] = originales[columna].astype(object)
        originales.loc[0, columna] = "not-a-number"
    elif cambio == "booleano":
        originales[columna] = True
    else:
        originales[columna] = originales[columna].astype(complex) + 1j
    monkeypatch.setattr(concrete.pd, "read_excel", lambda *args, **kwargs: originales)
    with pytest.raises(ValueError, match="ausentes|infinitos|numérica|reales"):
        concrete.preparar_concrete(b"source")


def test_cabeceras_documentadas_toleran_espacios_y_caso(lectura_xls):
    lectura_xls.columns = [
        f"  {' '.join(nombre.split()).upper()}  " for nombre in lectura_xls.columns
    ]
    archivos, metadata = concrete.preparar_concrete(b"original-xls-bytes")
    assert metadata["columnas"] == list(concrete.COLUMNAS_CONCRETE)
    assert metadata["original"]["columnas"] == lectura_xls.columns.tolist()
    assert len(pd.read_csv(BytesIO(archivos[concrete.ARCHIVO_CONCRETE]))) == 1028


def test_entrada_vacia_no_invoca_lector():
    with (
        patch.object(concrete.pd, "read_excel", side_effect=AssertionError("reader")),
        pytest.raises(ValueError, match="vacío"),
    ):
        concrete.preparar_concrete(b"")


def test_cli_idempotente_preserva_archivo_ajeno_y_valida_derivacion(
    tmp_path, lectura_xls
):
    origen = tmp_path / "fuente.xls"
    origen.write_bytes(b"original-xls-bytes")
    destino = tmp_path / "concrete"
    destino.mkdir()
    (destino / "README.md").write_text("keep", encoding="utf-8")
    args = crear_parser().parse_args([str(origen), "--salida", str(destino)])
    assert ejecutar(args) == destino
    antes = {
        ruta.name: (ruta.read_bytes(), ruta.stat().st_mtime_ns)
        for ruta in destino.iterdir()
    }
    assert ejecutar(args) == destino
    assert antes == {
        ruta.name: (ruta.read_bytes(), ruta.stat().st_mtime_ns)
        for ruta in destino.iterdir()
    }
    assert origen.read_bytes() == b"original-xls-bytes"
    metadata = json.loads((destino / "procedencia.json").read_text(encoding="utf-8"))
    concrete.validar_preparacion_concrete(destino, metadata)
    (destino / "concrete.csv").write_bytes(b"altered\n")
    with pytest.raises(ValueError, match="no coincide"):
        concrete.validar_preparacion_concrete(destino, metadata)
    metadata["filas"] = 999
    with pytest.raises(ValueError, match="procedencia"):
        concrete.validar_preparacion_concrete(destino, metadata)


@pytest.mark.parametrize(
    "nombre", ["Concrete_Data.xls", "concrete.csv", "procedencia.json"]
)
@pytest.mark.parametrize("directorio", [False, True])
def test_conflicto_en_cualquier_destino_no_crea_otros(
    tmp_path, lectura_xls, nombre, directorio
):
    origen = tmp_path / "fuente.xls"
    origen.write_bytes(b"original-xls-bytes")
    destino = tmp_path / "salida"
    destino.mkdir()
    conflicto = destino / nombre
    if directorio:
        conflicto.mkdir()
    else:
        conflicto.write_bytes(b"previous evidence\n")
    with pytest.raises(ValueError, match="archivo distinto"):
        ejecutar(argparse.Namespace(xls=origen, salida=destino))
    assert list(destino.iterdir()) == [conflicto]
    if not directorio:
        assert conflicto.read_bytes() == b"previous evidence\n"


def test_cli_exit_codes_y_salida_archivo(tmp_path, lectura_xls, capsys):
    origen = tmp_path / "fuente.xls"
    origen.write_bytes(b"original-xls-bytes")
    destino = tmp_path / "salida"
    assert main([str(origen), "--salida", str(destino)]) == 0
    assert "procedencia" in capsys.readouterr().out
    (destino / "procedencia.json").write_bytes(b"conflict")
    with pytest.raises(SystemExit) as salida:
        main([str(origen), "--salida", str(destino)])
    assert salida.value.code == 2
    assert "No se pudo preparar Concrete" in capsys.readouterr().err
    archivo = tmp_path / "not-a-directory"
    archivo.write_bytes(b"keep")
    with pytest.raises(ValueError, match="directorio"):
        ejecutar(argparse.Namespace(xls=origen, salida=archivo))
    assert archivo.read_bytes() == b"keep"
