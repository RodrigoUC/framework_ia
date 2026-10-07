"""Verifica exportaciones reproducibles sin usar los datasets académicos."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from scripts.ejecutar_lab02 import _json_seguro, crear_parser, ejecutar


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
        configuracion = json.loads((salida / "configuracion.json").read_text())
        self.assertEqual(
            configuracion["dataset"]["sha256"],
            hashlib.sha256(self.csv.read_bytes()).hexdigest(),
        )
        self.assertEqual(configuracion["dataset"]["filas"], 100)
        self.assertEqual(configuracion["dataset"]["fuente"], "fixture sintético")
        tabla = pd.read_csv(salida / "comparacion_validacion.csv")
        self.assertGreaterEqual(len(tabla), 4)
        resumen = json.loads((salida / "mejores_modelos.json").read_text())
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


if __name__ == "__main__":
    unittest.main()
