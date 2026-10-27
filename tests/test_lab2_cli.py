"""Verifica exportaciones reproducibles sin usar los datasets académicos."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from framework_ia.exportacion import (
    construir_archivos_experimento,
    construir_zip_experimento,
)
from framework_ia.modelos.supervisado import Clasificacion
from scripts.ejecutar_lab02 import _json_seguro, crear_parser, ejecutar, main


class Lab02CLITests(unittest.TestCase):
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
        return crear_parser().parse_args(
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
        salida = ejecutar(self.argumentos())
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
            self.assertEqual(list(predicciones.columns), ["fila", "real", "prediccion"])
            self.assertTrue(resumen[algoritmo]["parametros"])
            self.assertIn("accuracy", resumen[algoritmo]["metricas_test"])

    def test_repeticion_reproduce_resultados(self):
        uno = ejecutar(self.argumentos("uno"))
        dos = ejecutar(self.argumentos("dos"))
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
            ejecutar(argumentos)
        self.assertEqual(protegido.read_text(), "preservar")

    def test_target_invalido_no_crea_salida(self):
        argumentos = self.argumentos()
        argumentos.target = "ausente"
        with self.assertRaises((ValueError, KeyError)):
            ejecutar(argumentos)
        self.assertFalse(argumentos.salida.exists())

    def test_json_no_contiene_no_finitos(self):
        datos = _json_seguro({"a": np.nan, "b": np.float64(np.inf), "c": np.int64(2)})
        self.assertEqual(datos, {"a": None, "b": None, "c": 2})
        json.dumps(datos, allow_nan=False)

    def test_zip_del_experimento_equivale_a_cli_y_no_modifica_resultado(self):
        argumentos = self.argumentos()
        salida = ejecutar(argumentos)
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
        with patch(
            "framework_ia.exportacion.version", return_value="otra-version"
        ):
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
                "scripts.ejecutar_lab02.Clasificacion.experimentar",
                return_value=resultado,
            ),
            redirect_stdout(StringIO()) as mensajes,
        ):
            codigo = main(
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


if __name__ == "__main__":
    unittest.main()
