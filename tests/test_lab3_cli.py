"""Local regression export/CLI checks; fixtures are not academic evidence."""

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from framework_ia import Regresion
from framework_ia.datos.fuentes import CargadorCSV
from framework_ia.exportacion import (
    construir_archivos_experimento,
    construir_archivos_modelo,
    construir_zip_experimento,
)
from framework_ia.utils import ConfiguracionCSV
from scripts.ejecutar_lab03 import crear_parser, ejecutar, main


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(15)
    data = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    data["target"] = 3 * data.a - data.b + rng.normal(size=len(data))
    path = tmp_path / "observaciones.csv"
    path.write_text(data.to_csv(index=False, sep=";", decimal=","), encoding="utf-8")
    raw = path.read_bytes()
    loaded = CargadorCSV.cargar_bytes(raw, ConfiguracionCSV(separador=";", decimal=","))
    return path, raw, loaded


def arguments(source, tmp_path, algorithms=("RLS", "RLM", "LassoCV", "RidgeCV")):
    return crear_parser().parse_args(
        [
            str(source[0]),
            "--target",
            "target",
            "--salida",
            str(tmp_path / "evidence"),
            "--separador",
            ";",
            "--decimal",
            ",",
            "--folds",
            "3",
            "--feature-simple",
            "b",
            "--fuente",
            "synthetic software fixture",
            "--algoritmos",
            *algorithms,
        ]
    )


def provenance(source):
    path, raw, data = source
    return {
        "archivo": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "filas": len(data),
        "columnas": list(data.columns),
        "fuente": "synthetic software fixture",
        "lectura": {"separador": ";", "decimal": ",", "encoding": "utf-8"},
    }


def test_cli_bytes_equal_shared_exporter_and_preserve_original_csv_hash(
    source, tmp_path
):
    args = arguments(source, tmp_path)
    captured = []
    original = Regresion.experimentar

    def experiment(model, **options):
        result = original(model, **options)
        captured.append(result)
        return result

    with patch.object(Regresion, "experimentar", autospec=True, side_effect=experiment):
        output = ejecutar(args)
    result = captured[0]
    expected = construir_archivos_experimento(
        result, argumentos=vars(args), dataset=provenance(source)
    )
    actual = {
        str(path.relative_to(output)).replace("\\", "/"): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    config = json.loads(actual["configuracion.json"])
    assert config["dataset"]["sha256"] == hashlib.sha256(source[1]).hexdigest()
    assert config["dataset"]["lectura"]["decimal"] == ","
    assert config["experimento"]["test_usado_para_seleccion"] is False
    summary = json.loads(actual["mejores_modelos.json"])
    assert summary["RLS"]["features"] == ["b"]
    assert isinstance(summary["LassoCV"]["parametros"]["cv"], dict)
    assert summary["LassoCV"]["metadatos"]["alpha_elegido"] > 0
    predictions = pd.read_csv(
        BytesIO(actual["RLM/predicciones_test.csv"]), index_col="fila"
    )
    np.testing.assert_allclose(
        predictions.residuo, predictions.real - predictions.prediccion
    )
    assert predictions.index.tolist() == result.metadatos["test_indices"]
    assert not any("matriz_confusion" in name for name in actual)


def test_individual_and_experiment_exports_share_bytes_and_never_retrain_or_mutate(
    source,
):
    result = Regresion(dataframe=source[2]).experimentar(algoritmos="RidgeCV", folds=3)
    winner = result.mejores["RidgeCV"]
    table_before = result.tabla.copy(deep=True)
    predictions_before = winner.y_pred.copy()
    inputs = {"argumentos": {"target": "target"}, "dataset": provenance(source)}
    with patch.object(
        Regresion, "_ajustar", side_effect=AssertionError("export must not fit")
    ):
        individual = construir_archivos_modelo(winner, **inputs)
        experiment = construir_archivos_experimento(result, **inputs)
        archive = construir_zip_experimento(experiment)
        assert archive == construir_zip_experimento(experiment)
    assert set(individual) == {
        "configuracion.json",
        "metricas.json",
        "predicciones_train.csv",
        "predicciones_test.csv",
    }
    for name, content in individual.items():
        assert content == experiment[f"RidgeCV/{name}"]
    with ZipFile(BytesIO(archive)) as zipped:
        assert {name: zipped.read(name) for name in zipped.namelist()} == experiment
    assert_frame_equal(table_before, result.tabla)
    assert_series_equal(predictions_before, winner.y_pred)
    individual_config = json.loads(individual["configuracion.json"])
    assert individual_config["modelo"]["algoritmo"] == "RidgeCV"
    assert individual_config["dataset"] == provenance(source)


def test_stored_versions_override_export_time_versions_and_json_is_finite(source):
    result = Regresion(dataframe=source[2]).experimentar(algoritmos="Ridge", folds=3)
    metadata = {
        **result.metadatos,
        "versiones": {"python": "recorded-python", "scikit-learn": "recorded-sklearn"},
        "nonfinite": np.float64(np.inf),
    }
    saved = replace(result, metadatos=metadata)
    files = construir_archivos_experimento(
        saved,
        argumentos={},
        dataset=provenance(source),
        versiones={
            "python": "current-python",
            "scikit-learn": "current-sklearn",
            "streamlit": "extra",
        },
    )
    config = json.loads(files["configuracion.json"])
    assert config["versiones"] == {
        "python": "recorded-python",
        "scikit-learn": "recorded-sklearn",
        "streamlit": "extra",
    }
    assert config["experimento"]["nonfinite"] is None
    for name, data in files.items():
        if name.endswith(".json"):
            json.loads(
                data,
                parse_constant=lambda value: pytest.fail(f"Nonfinite JSON: {value}"),
            )


def test_cli_rejects_nonempty_output_and_invalid_target_without_artifacts(
    source, tmp_path
):
    args = arguments(source, tmp_path, algorithms=("Ridge",))
    args.target = "missing"
    with pytest.raises(KeyError):
        ejecutar(args)
    assert not args.salida.exists()
    args.salida.mkdir()
    protected = args.salida / "existing.txt"
    protected.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="carpeta nueva"):
        ejecutar(args)
    assert protected.read_text(encoding="utf-8") == "keep"


def test_cli_exit_codes_complete_partial_and_invalid(source, tmp_path, capsys):
    complete = [
        str(source[0]),
        "--target",
        "target",
        "--separador",
        ";",
        "--decimal",
        ",",
        "--algoritmos",
        "Ridge",
        "--salida",
        str(tmp_path / "complete"),
    ]
    assert main(complete) == 0
    partial = Regresion(dataframe=source[2]).experimentar(
        algoritmos=["DT", "Ridge"],
        folds=3,
        configuraciones={"DT": [{"nombre": "invalid", "parametros": {"max_depth": 0}}]},
    )
    with patch.object(Regresion, "experimentar", return_value=partial):
        assert main([*complete[:-1], str(tmp_path / "partial")]) == 1
    assert "Ejecución parcial" in capsys.readouterr().out
    all_failed = Regresion(dataframe=source[2]).experimentar(
        algoritmos="DT",
        folds=3,
        configuraciones={"DT": [{"nombre": "invalid", "parametros": {"max_depth": 0}}]},
    )
    with (
        patch.object(Regresion, "experimentar", return_value=all_failed),
        pytest.raises(SystemExit) as failure,
    ):
        main([*complete[:-1], str(tmp_path / "failed")])
    assert failure.value.code == 2
    assert not (tmp_path / "failed").exists()


def test_exporter_supports_a_real_individual_result_without_experiment(source):
    winner = Regresion(dataframe=source[2], features=["a"]).RLS()
    files = construir_archivos_modelo(winner, argumentos={}, dataset=provenance(source))
    config = json.loads(files["configuracion.json"])
    metrics = json.loads(files["metricas.json"])
    assert config["modelo"]["features"] == ["a"]
    assert metrics == {"train": winner.metricas_train, "test": winner.metricas}
    assert "experimento" not in config
