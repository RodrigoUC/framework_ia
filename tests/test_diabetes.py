"""Pruebas unificadas de preparación, trazabilidad y referencia de Diabetes."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from framework_ia.datasets.diabetes import (
    COLUMNAS_ORIGINALES,
    PREDICTORAS_DIABETES,
    cargar_diabetes_regresion,
    preparar_diabetes,
    validar_preparacion_diabetes,
)
from framework_ia.modelos.supervisado import Regresion
from framework_ia.ui import streamlit_app
from framework_ia.ui.controladores.controlador_regresion import snapshot_ejecucion
from scripts.preparar_diabetes import crear_parser, ejecutar, main

REFERENCIA = "Diabetes · referencia Lab03"
ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


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


@pytest.mark.parametrize(
    "cambio",
    [
        "id_duplicado",
        "glyhb_texto",
        "glyhb_infinito",
        "sin_objetivo",
        "columna_ausente",
    ],
)
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


def test_reference_is_offline_raw_fresh_and_deterministic():
    with patch("socket.create_connection", side_effect=AssertionError("network")):
        datos, contenido, metadata = cargar_diabetes_regresion()
        repetido, bytes_repetidos, metadata_repetida = cargar_diabetes_regresion()
    assert datos.shape == (442, 11)
    assert datos.columns.tolist() == [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "target",
    ]
    assert datos.loc[0, "bmi"] == 32.1
    assert (datos.target.min(), datos.target.max()) == (25, 346)
    assert not datos.isna().any().any()
    assert contenido == bytes_repetidos
    assert metadata == metadata_repetida
    assert metadata["sha256"] == hashlib.sha256(contenido).hexdigest()
    assert metadata["hash_sobre"] == "csv_materializado"
    assert metadata["carga"] == {
        "funcion": "load_diabetes",
        "as_frame": True,
        "scaled": False,
    }
    assert metadata["preset_regresion"]["feature_simple"] == "bmi"
    pd.testing.assert_frame_equal(pd.read_csv(BytesIO(contenido)), datos)
    datos.loc[0, "bmi"] = -999
    assert repetido.loc[0, "bmi"] == 32.1
    assert cargar_diabetes_regresion()[0].loc[0, "bmi"] == 32.1


def test_preparation_is_idempotent_and_preserves_unrelated_files(tmp_path):
    destino = tmp_path / "referencia"
    destino.mkdir()
    (destino / "README.md").write_text("existing", encoding="utf-8")
    args = crear_parser().parse_args(["--salida", str(destino)])
    assert ejecutar(args) == destino
    antes = {
        ruta.name: (ruta.read_bytes(), ruta.stat().st_mtime_ns)
        for ruta in destino.iterdir()
    }
    ejecutar(args)
    assert antes == {
        ruta.name: (ruta.read_bytes(), ruta.stat().st_mtime_ns)
        for ruta in destino.iterdir()
    }
    metadata = json.loads((destino / "procedencia.json").read_text(encoding="utf-8"))
    assert (
        metadata["sha256"]
        == hashlib.sha256((destino / metadata["archivo"]).read_bytes()).hexdigest()
    )
    assert set(metadata["versiones_origen"]) == {
        "python",
        "numpy",
        "pandas",
        "scikit-learn",
    }


@pytest.mark.parametrize("nombre", ["diabetes_regresion.csv", "procedencia.json"])
@pytest.mark.parametrize("directorio", [False, True])
def test_all_destinations_are_preflighted_before_any_write(
    tmp_path, nombre, directorio
):
    ruta = tmp_path / nombre
    if directorio:
        ruta.mkdir()
    else:
        ruta.write_bytes(b"previous evidence\n")
    with pytest.raises(ValueError, match="archivo distinto"):
        ejecutar(argparse.Namespace(salida=tmp_path))
    assert [archivo.name for archivo in tmp_path.iterdir()] == [nombre]
    if not directorio:
        assert ruta.read_bytes() == b"previous evidence\n"


def test_cli_success_and_conflict_exit_codes(tmp_path, capsys):
    assert main(["--salida", str(tmp_path)]) == 0
    assert "procedencia" in capsys.readouterr().out
    (tmp_path / "procedencia.json").write_text("invalid", encoding="utf-8")
    with pytest.raises(SystemExit) as salida:
        main(["--salida", str(tmp_path)])
    assert salida.value.code == 2
    assert "No se pudo preparar" in capsys.readouterr().err


def nueva_app():
    datos = pd.DataFrame(
        {"a": range(30), "b": range(30, 60), "respuesta": range(60, 90)}
    )
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (datos, "arbitrario.csv", "arbitrario")
    app.run()
    assert not app.exception
    return app


def aplicar_referencia(app):
    app.sidebar.radio(key="fuente_origen").set_value(REFERENCIA).run()
    app.sidebar.button(
        key="FormSubmitter:form_fuente_datos-Aplicar fuente"
    ).click().run()
    assert not app.exception


def test_source_selection_only_applies_on_confirmation_and_never_trains():
    app = nueva_app()
    app.session_state["resultado_lab3_modelo_RLM"] = {"stale": True}
    with (
        patch.object(Regresion, "entrenar", side_effect=AssertionError("auto-fit")),
        patch.object(Regresion, "experimentar", side_effect=AssertionError("auto-fit")),
    ):
        app.sidebar.radio(key="fuente_origen").set_value(REFERENCIA).run()
        assert app.session_state["fuente_aplicada"][2] == "arbitrario"
        assert "resultado_lab3_modelo_RLM" in app.session_state
        app.sidebar.button(
            key="FormSubmitter:form_fuente_datos-Aplicar fuente"
        ).click().run()
        assert not app.exception
        assert app.session_state["dataset_preparado"].shape == (442, 11)
        assert app.session_state["sup_target"] == "target"
        assert app.session_state["sup_feature_simple"] == "bmi"
        assert app.session_state["sup_test_size"] == 25
        assert app.session_state["sup_random_state"] == 42
        assert len(app.session_state["sup_features"]) == 10
        assert "resultado_lab3_modelo_RLM" not in app.session_state
        assert "fuente_preset_regresion" not in app.session_state
        app.sidebar.button(key="nav_configuracion").click().run()
        assert not app.exception
        assert app.selectbox(key="sup_feature_simple").value == "bmi"
        app.selectbox(key="sup_feature_simple").set_value("bp").run()
        app.number_input(key="sup_random_state").set_value(17).run()
        assert app.sidebar.button(key="nav_lab3_resultados").disabled
        app.sidebar.button(key="nav_eda").click().run()
        app.sidebar.button(key="nav_configuracion").click().run()
        assert not app.exception
    assert app.selectbox(key="sup_feature_simple").value == "bp"
    assert app.number_input(key="sup_random_state").value == 17


def test_reference_failure_preserves_source_and_csv_remains_available():
    app = nueva_app()
    app.sidebar.radio(key="fuente_origen").set_value(REFERENCIA).run()
    with patch.object(
        streamlit_app,
        "cargar_diabetes_regresion",
        side_effect=ValueError("fixture failure"),
    ):
        app.sidebar.button(
            key="FormSubmitter:form_fuente_datos-Aplicar fuente"
        ).click().run()
    assert not app.exception
    assert app.session_state["fuente_aplicada"][2] == "arbitrario"
    assert any("fixture failure" in aviso.value for aviso in app.error)
    app.sidebar.radio(key="fuente_origen").set_value("Subir CSV").run()
    assert app.get("file_uploader")
    app.sidebar.radio(key="fuente_origen").set_value("CSV local").run()
    app.sidebar.button(
        key="FormSubmitter:form_fuente_datos-Aplicar fuente"
    ).click().run()
    assert not app.exception
    assert app.session_state["fuente_aplicada"][2].startswith("local:")
    assert app.session_state["sup_tarea"] is None
    assert app.session_state["sup_feature_simple"] != "bmi"


def test_pending_preset_is_identity_scoped_and_export_keeps_materialization():
    estado = {"fuente_preset_regresion": {"identidad": "old"}, "sup_target": "manual"}
    with patch.object(streamlit_app.st, "session_state", estado):
        streamlit_app._aplicar_preset_regresion("new")
    assert estado == {"sup_target": "manual"}
    datos, _, metadata = cargar_diabetes_regresion()
    estado = {"fuente_metadata": metadata}
    with patch.object(streamlit_app.st, "session_state", estado):
        snapshot = snapshot_ejecucion(None, datos, {}, {})
    assert snapshot["dataset"]["hash_sobre"] == "csv_materializado"
    assert snapshot["dataset"]["carga"]["scaled"] is False
    metadata["carga"]["scaled"] = True
    assert snapshot["dataset"]["carga"]["scaled"] is False
