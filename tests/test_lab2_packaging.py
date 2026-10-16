"""Valida integridad y portabilidad del paquete con un experimento sintético real.

Los datos son sintéticos: verifican el software, no son resultados académicos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from framework_ia.datos.diabetes_lab02 import COLUMNAS_ORIGINALES, PREDICTORAS_DIABETES
from framework_ia.datos.potabilidad_lab02 import PREDICTORAS_POTABILIDAD
from scripts import ejecutar_lab02, preparar_diabetes_lab02, preparar_potabilidad_lab02
from scripts.preparar_entrega_lab02 import crear_parser, ejecutar

RAIZ_REAL = Path(__file__).resolve().parents[1]


def _potabilidad_sintetica() -> bytes:
    rng = np.random.default_rng(1)
    datos = pd.DataFrame(
        rng.normal(size=(90, len(PREDICTORAS_POTABILIDAD))),
        columns=list(PREDICTORAS_POTABILIDAD),
    )
    datos["Potability"] = (datos["ph"] + datos["Sulfate"] > 0).astype(int)
    datos.loc[[0, 5, 9], "ph"] = np.nan
    return datos.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _diabetes_sintetica() -> bytes:
    rng = np.random.default_rng(2)
    filas = 90
    datos = pd.DataFrame(
        {columna: rng.normal(100, 10, filas).round(1) for columna in COLUMNAS_ORIGINALES}
    )
    datos["id"] = range(1000, 1000 + filas)
    datos["glyhb"] = np.where(datos["stab.glu"] > 105, 8.0, 5.0)
    datos.loc[[3, 4], "glyhb"] = np.nan
    datos["location"] = rng.choice(["Buckingham", "Louisa"], filas)
    datos["gender"] = rng.choice(["female", "male"], filas)
    datos["frame"] = rng.choice(["small", "medium", "large"], filas)
    datos = datos[list(COLUMNAS_ORIGINALES)]
    return datos.to_csv(index=False, lineterminator="\n").encode("utf-8")


class Lab02PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporal = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.temporal.name)
        for nombre in (
            "app.py",
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-lab02-lock.txt",
        ):
            (cls.raiz / nombre).write_text("# fixture sintético\n", encoding="utf-8")
        (cls.raiz / ".streamlit").mkdir()
        (cls.raiz / ".streamlit" / "config.toml").write_text(
            '[theme]\nbase = "dark"\n', encoding="utf-8"
        )
        documentos = cls.raiz / "docs" / "lab02"
        documentos.mkdir(parents=True)
        (documentos / "LEEME_ENTREGA.md").write_text(
            "Abrir la aplicación desde la carpeta extraída con streamlit run app.py\n",
            encoding="utf-8",
        )
        for carpeta in ("framework_ia", "scripts", "tests"):
            (cls.raiz / carpeta).mkdir()
            (cls.raiz / carpeta / "fixture.py").write_text("# código\n", encoding="utf-8")
        for carpeta in (".git", "env", "output"):
            (cls.raiz / carpeta).mkdir()
            (cls.raiz / carpeta / "excluir.txt").write_text("no", encoding="utf-8")
        cls.datos = cls.raiz / "data" / "lab02"
        cls.datos.mkdir(parents=True)
        entradas = cls.raiz / "entradas"
        entradas.mkdir()
        (entradas / "water_potability.csv").write_bytes(_potabilidad_sintetica())
        (entradas / "diabetes_original.csv").write_bytes(_diabetes_sintetica())
        preparar_potabilidad_lab02.ejecutar(
            argparse.Namespace(csv=entradas / "water_potability.csv", salida=cls.datos)
        )
        preparar_diabetes_lab02.ejecutar(
            argparse.Namespace(csv=entradas / "diabetes_original.csv", salida=cls.datos)
        )
        procedencia = json.loads((cls.datos / "procedencia.json").read_text(encoding="utf-8"))
        cls.resultados = cls.raiz / "resultados"
        comunes = ["--folds", "2", "--semilla", "3"]
        ejecutar_lab02.main(
            [
                str(cls.datos / "potabilidad.csv"), "--target", "Potability",
                "--salida", str(cls.resultados / "potabilidad"),
                "--fuente", procedencia["potabilidad"]["fuente"], *comunes,
            ]
        )
        ejecutar_lab02.main(
            [
                str(cls.datos / "diabetes.csv"), "--target", "diabetes",
                "--features", *PREDICTORAS_DIABETES, "--incluir-categoricas",
                "--salida", str(cls.resultados / "diabetes"),
                "--fuente", procedencia["diabetes"]["fuente"], *comunes,
            ]
        )
        informe = cls.raiz / "fuentes"
        informe.mkdir()
        (informe / "main.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (informe / "main.tex").write_text(
            "\\documentclass{article}\n\\begin{document}Fixture\\end{document}\n",
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporal.cleanup()

    def argumentos(self, salida="entrega", resultados=None):
        return crear_parser().parse_args(
            [
                "--ids", "1", "2", "3", "4",
                "--informe", str(self.raiz / "fuentes" / "main.pdf"),
                "--fuentes-tex", str(self.raiz / "fuentes" / "main.tex"),
                "--resultados", str(resultados or self.resultados),
                "--salida", str(self.raiz / salida),
            ]
        )

    def test_entrega_repetible_manifest_verificable_y_cuatro_ids(self):
        uno = ejecutar(self.argumentos("uno"), raiz=self.raiz)
        dos = ejecutar(self.argumentos("dos"), raiz=self.raiz)
        self.assertEqual(uno.name, "1_2_3_4.zip")
        self.assertEqual(uno.read_bytes(), dos.read_bytes())
        with ZipFile(BytesIO(uno.read_bytes())) as paquete:
            nombres = set(paquete.namelist())
            for requerido in (
                "informe/main.pdf",
                "informe/main.tex",
                ".streamlit/config.toml",
                "requirements-lab02-lock.txt",
                "salida_lab02/diabetes/DT/arbol.txt",
                "salida_lab02/potabilidad/ranking_atributos.csv",
            ):
                self.assertIn(requerido, nombres)
            csv_datos = {
                nombre for nombre in nombres
                if nombre.startswith("data/lab02/") and nombre.endswith(".csv")
            }
            self.assertEqual(
                csv_datos,
                {
                    "data/lab02/water_potability.csv",
                    "data/lab02/potabilidad.csv",
                    "data/lab02/diabetes_original.csv",
                    "data/lab02/diabetes.csv",
                    "data/lab02/trazabilidad_diabetes.csv",
                },
            )
            self.assertIn(b"streamlit run app.py", paquete.read("README.md"))
            for excluido in ("env/excluir.txt", ".git/excluir.txt", "output/excluir.txt"):
                self.assertNotIn(excluido, nombres)
            manifiesto = json.loads(paquete.read("MANIFIESTO.json"))
            self.assertEqual(set(manifiesto["archivos"]), nombres - {"MANIFIESTO.json"})
            self.assertNotIn(str(self.raiz), paquete.read("MANIFIESTO.json").decode("utf-8"))
            for nombre, registro in manifiesto["archivos"].items():
                contenido = paquete.read(nombre)
                self.assertEqual(registro["bytes"], len(contenido))
                self.assertEqual(registro["sha256"], hashlib.sha256(contenido).hexdigest())

    def _copia_resultados(self):
        destino = Path(tempfile.mkdtemp(dir=self.raiz))
        shutil.copytree(self.resultados, destino, dirs_exist_ok=True)
        return destino

    def test_csv_preparado_modificado_rechaza_entrega(self):
        csv = self.datos / "potabilidad.csv"
        previo = csv.read_bytes()
        try:
            csv.write_bytes(previo + b"1,1,1,1,1,1,1,1,1,0\n")
            argumentos = self.argumentos("modificado")
            with self.assertRaises(ValueError):
                ejecutar(argumentos, raiz=self.raiz)
            self.assertFalse(argumentos.salida.exists())
        finally:
            csv.write_bytes(previo)

    def test_predictora_del_target_o_accuracy_incoherente_rechazan_entrega(self):
        copia = self._copia_resultados()
        ruta = copia / "diabetes" / "configuracion.json"
        configuracion = json.loads(ruta.read_text(encoding="utf-8"))
        configuracion["argumentos"]["features"].append("glyhb")
        ruta.write_text(json.dumps(configuracion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "17 predictoras"):
            ejecutar(self.argumentos("x1", copia), raiz=self.raiz)
        copia = self._copia_resultados()
        predicciones = copia / "potabilidad" / "DT" / "predicciones_test.csv"
        tabla = pd.read_csv(predicciones)
        tabla.loc[0, "prediccion"] = 1 - tabla.loc[0, "prediccion"]
        tabla.to_csv(predicciones, index=False)
        with self.assertRaisesRegex(ValueError, "accuracy"):
            ejecutar(self.argumentos("x2", copia), raiz=self.raiz)

    def test_evidencia_faltante_rechaza_entrega(self):
        copia = self._copia_resultados()
        (copia / "diabetes" / "DT" / "arbol.txt").unlink()
        with self.assertRaisesRegex(ValueError, "evidencia incompleta"):
            ejecutar(self.argumentos("x3", copia), raiz=self.raiz)

    def test_pdf_sin_firma_o_latex_incompleto_rechaza_entrega(self):
        argumentos = self.argumentos("x4")
        pdf = argumentos.informe.read_bytes()
        tex = argumentos.fuentes_tex.read_text(encoding="utf-8")
        try:
            argumentos.informe.write_text("no es un PDF", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "PDF"):
                ejecutar(argumentos, raiz=self.raiz)
            argumentos.informe.write_bytes(pdf)
            argumentos.fuentes_tex.write_text("texto incompleto", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LaTeX completo"):
                ejecutar(argumentos, raiz=self.raiz)
            self.assertFalse(argumentos.salida.exists())
        finally:
            argumentos.informe.write_bytes(pdf)
            argumentos.fuentes_tex.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
