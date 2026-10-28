"""Pruebas unificadas de exportación y CLI de clasificación y regresión."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.datasets import make_classification

from framework_ia import Regresion
from framework_ia.datos.fuentes import CargadorCSV
from framework_ia.exportacion import (
    construir_archivos_experimento,
    construir_archivos_modelo,
    construir_zip_experimento,
)
from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.utils import ConfiguracionCSV
from scripts import ejecutar_clasificacion, ejecutar_regresion


class ClasificacionCLITests(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        x, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=7,
        )
        self.csv = self.raiz / "fixture_sintetico.csv"
        datos = pd.DataFrame(x, columns=["a", "b", "c", "d"])
        datos["clase"] = y
        datos.to_csv(self.csv, index=False)

    def argumentos(self, nombre="resultados"):
        return ejecutar_clasificacion.crear_parser().parse_args(
            [
                str(self.csv),
                "--target",
                "clase",
                "--algoritmos",
                "KNN",
                "DT",
                "--salida",
                str(self.raiz / nombre),
                "--fuente",
                "fixture sintético",
            ]
        )

    def test_exporta_hash_parametros_y_test_separado(self):
        salida = ejecutar_clasificacion.ejecutar(self.argumentos())
        configuracion = json.loads(
            (salida / "configuracion.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            configuracion["dataset"]["sha256"],
            hashlib.sha256(self.csv.read_bytes()).hexdigest(),
        )
        self.assertEqual(configuracion["dataset"]["filas"], 100)
        self.assertEqual(configuracion["dataset"]["fuente"], "fixture sintético")
        tabla = pd.read_csv(salida / "comparacion_validacion.csv")
        self.assertGreaterEqual(len(tabla), 4)
        resumen = json.loads(
            (salida / "mejores_modelos.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(resumen), {"KNN", "DT"})
        for algoritmo in resumen:
            predicciones = pd.read_csv(salida / algoritmo / "predicciones_test.csv")
            self.assertEqual(len(predicciones), 25)
            self.assertEqual(
                list(predicciones.columns), ["fila", "real", "prediccion"]
            )
            self.assertTrue(resumen[algoritmo]["parametros"])
            self.assertIn("accuracy", resumen[algoritmo]["metricas_test"])

    def test_repeticion_reproduce_resultados(self):
        uno = ejecutar_clasificacion.ejecutar(self.argumentos("uno"))
        dos = ejecutar_clasificacion.ejecutar(self.argumentos("dos"))
        self.assertEqual(
            (uno / "comparacion_validacion.csv").read_bytes(),
            (dos / "comparacion_validacion.csv").read_bytes(),
        )
        self.assertEqual(
            (uno / "mejores_modelos.json").read_bytes(),
            (dos / "mejores_modelos.json").read_bytes(),
        )

    def test_no_sobrescribe_directorio_no_vacio(self):
        argumentos = self.argumentos()
        argumentos.salida.mkdir()
        protegido = argumentos.salida / "nota.txt"
        protegido.write_text("preservar")
        with self.assertRaisesRegex(ValueError, "carpeta nueva"):
            ejecutar_clasificacion.ejecutar(argumentos)
        self.assertEqual(protegido.read_text(), "preservar")

    def test_target_invalido_no_crea_salida(self):
        argumentos = self.argumentos()
        argumentos.target = "ausente"
        with self.assertRaises((ValueError, KeyError)):
            ejecutar_clasificacion.ejecutar(argumentos)
        self.assertFalse(argumentos.salida.exists())

    def test_json_no_contiene_no_finitos(self):
        datos = ejecutar_clasificacion._json_seguro(
            {"a": np.nan, "b": np.float64(np.inf), "c": np.int64(2)}
        )
        self.assertEqual(datos, {"a": None, "b": None, "c": 2})
        json.dumps(datos, allow_nan=False)

    def test_zip_del_experimento_equivale_a_cli_y_no_modifica_resultado(self):
        argumentos = self.argumentos()
        salida = ejecutar_clasificacion.ejecutar(argumentos)
        datos = pd.read_csv(self.csv)
        resultado = Clasificacion(dataframe=datos, target="clase").experimentar(
            algoritmos=argumentos.algoritmos,
            test_size=argumentos.test_size,
            random_state=argumentos.semilla,
            estrategia=argumentos.estrategia,
            folds=argumentos.folds,
            seleccion_k=argumentos.seleccion_k,
        )
        tabla_original = resultado.tabla.copy(deep=True)
        metadatos_originales = deepcopy(resultado.metadatos)
        dataset = {
            "archivo": self.csv.name,
            "sha256": hashlib.sha256(self.csv.read_bytes()).hexdigest(),
            "filas": len(datos),
            "columnas": list(datos.columns),
            "fuente": "fixture sintético",
        }
        archivos = construir_archivos_experimento(
            resultado, argumentos=vars(argumentos), dataset=dataset
        )
        contenido_zip = construir_zip_experimento(archivos)
        archivos_cli = {
            ruta.relative_to(salida).as_posix(): ruta.read_bytes()
            for ruta in salida.rglob("*")
            if ruta.is_file()
        }
        with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
            self.assertEqual(set(archivo_zip.namelist()), set(archivos_cli))
            for nombre, contenido in archivos_cli.items():
                self.assertEqual(archivo_zip.read(nombre), contenido, nombre)
            resumen = json.loads(archivo_zip.read("mejores_modelos.json"))
            for algoritmo in ("KNN", "DT"):
                predicciones = pd.read_csv(
                    BytesIO(archivo_zip.read(f"{algoritmo}/predicciones_test.csv"))
                )
                accuracy_observada = (
                    predicciones.real == predicciones.prediccion
                ).mean()
                self.assertAlmostEqual(
                    resumen[algoritmo]["metricas_test"]["accuracy"],
                    accuracy_observada,
                )
        self.assertEqual(construir_zip_experimento(archivos), contenido_zip)
        pd.testing.assert_frame_equal(resultado.tabla, tabla_original)
        self.assertEqual(resultado.metadatos, metadatos_originales)

    def test_exportacion_conserva_procedencia_y_versiones_del_ajuste(self):
        resultado = Clasificacion(
            dataframe=pd.read_csv(self.csv), target="clase"
        ).experimentar(algoritmos=["DT"])
        versiones_originales = dict(resultado.metadatos["versiones"])
        dataset = {
            "archivo": self.csv.name,
            "sha256": hashlib.sha256(self.csv.read_bytes()).hexdigest(),
            "fuente": "fixture sintético",
            "lectura": {"separador": ",", "decimal": ".", "usar_indice": False},
        }
        with patch("framework_ia.exportacion.version", return_value="otra-version"):
            archivos = construir_archivos_experimento(
                resultado, argumentos={"semilla": 42}, dataset=dataset
            )
        configuracion = json.loads(archivos["configuracion.json"])
        self.assertEqual(configuracion["dataset"], dataset)
        for paquete, registrada in versiones_originales.items():
            self.assertEqual(configuracion["versiones"][paquete], registrada)
        self.assertEqual(dataset["lectura"]["usar_indice"], False)

    def test_cli_parcial_conserva_error_y_solo_exporta_ganadores_validos(self):
        resultado = Clasificacion(
            dataframe=pd.read_csv(self.csv), target="clase"
        ).experimentar(
            algoritmos=["KNN", "DT"],
            configuraciones={
                "KNN": [{"nombre": "inviable", "parametros": {"n_neighbors": 10000}}]
            },
        )
        argumentos = self.argumentos()
        with (
            patch(
                "scripts.ejecutar_clasificacion.Clasificacion.experimentar",
                return_value=resultado,
            ),
            redirect_stdout(StringIO()) as mensajes,
        ):
            codigo = ejecutar_clasificacion.main(
                [
                    str(self.csv),
                    "--target",
                    "clase",
                    "--algoritmos",
                    "KNN",
                    "DT",
                    "--salida",
                    str(argumentos.salida),
                ]
            )
        self.assertEqual(codigo, 1)
        self.assertIn("Ejecución parcial", mensajes.getvalue())
        comparacion = pd.read_csv(argumentos.salida / "comparacion_validacion.csv")
        self.assertEqual(comparacion.iloc[0].estado, "error")
        self.assertIn("ValueError", comparacion.iloc[0].error)
        resumen = json.loads((argumentos.salida / "mejores_modelos.json").read_bytes())
        self.assertEqual(set(resumen), {"DT"})
        self.assertFalse((argumentos.salida / "KNN").exists())


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
    return ejecutar_regresion.crear_parser().parse_args(
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
        output = ejecutar_regresion.ejecutar(args)
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
        ejecutar_regresion.ejecutar(args)
    assert not args.salida.exists()
    args.salida.mkdir()
    protected = args.salida / "existing.txt"
    protected.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="carpeta nueva"):
        ejecutar_regresion.ejecutar(args)
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
    assert ejecutar_regresion.main(complete) == 0
    partial = Regresion(dataframe=source[2]).experimentar(
        algoritmos=["DT", "Ridge"],
        folds=3,
        configuraciones={"DT": [{"nombre": "invalid", "parametros": {"max_depth": 0}}]},
    )
    with patch.object(Regresion, "experimentar", return_value=partial):
        assert ejecutar_regresion.main([*complete[:-1], str(tmp_path / "partial")]) == 1
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
        ejecutar_regresion.main([*complete[:-1], str(tmp_path / "failed")])
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
