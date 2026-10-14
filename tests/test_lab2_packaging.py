"""Valida integridad y portabilidad del paquete usando evidencia sintética."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from framework_ia.utils import ALGORITMOS_LAB2
from scripts.preparar_entrega_lab02 import crear_parser, ejecutar


class Lab02PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        for nombre in (
            "app.py",
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-lab02-lock.txt",
        ):
            (self.raiz / nombre).write_text("# fixture sintético\n", encoding="utf-8")
        (self.raiz / ".streamlit").mkdir()
        (self.raiz / ".streamlit" / "config.toml").write_text(
            '[theme]\nbase = "dark"\n', encoding="utf-8"
        )
        instrucciones = self.raiz / "docs" / "lab02"
        instrucciones.mkdir(parents=True)
        (instrucciones / "LEEME_ENTREGA.md").write_text(
            "Abrir la aplicación desde la carpeta extraída con streamlit run app.py\n",
            encoding="utf-8",
        )
        (self.raiz / "README.md").write_text("cd framework_start\n", encoding="utf-8")
        for carpeta in ("framework_ia", "scripts", "tests"):
            (self.raiz / carpeta).mkdir()
            (self.raiz / carpeta / "fixture.py").write_text(
                "# Código sintético para verificar el inventario del paquete\n",
                encoding="utf-8",
            )
        datos = self.raiz / "data" / "lab02"
        datos.mkdir(parents=True)
        (datos / "procedencia.json").write_text(
            '{"tipo":"fixture sintético"}', encoding="utf-8"
        )
        for nombre, valores in (
            ("potabilidad", [1, 2, 3, 4]),
            ("diabetes", [11, 12, 13, 14]),
        ):
            csv = datos / f"{nombre}.csv"
            pd.DataFrame({"x": valores, "target": [0, 1, 0, 1]}).to_csv(
                csv, index=False
            )
            resultados = self.raiz / "resultados" / nombre
            resultados.mkdir(parents=True)
            configuracion = {
                "dataset": {
                    "archivo": csv.name,
                    "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                },
                "experimento": {
                    "candidatos_fallidos": 0,
                    "candidatos_exitosos": 15,
                    "errores_prueba": {},
                    "test_usado_para_seleccion": False,
                },
            }
            (resultados / "configuracion.json").write_text(
                json.dumps(configuracion), encoding="utf-8"
            )
            comparacion = pd.DataFrame(
                [
                    {
                        "algoritmo": algoritmo,
                        "configuracion": variante,
                        "estado": "ok",
                        "seleccionado": variante == "estandar",
                    }
                    for algoritmo in ALGORITMOS_LAB2
                    for variante in ("estandar", "variante_a", "variante_b")
                ]
            )
            comparacion.to_csv(resultados / "comparacion_validacion.csv", index=False)
            mejores = {}
            for algoritmo in ALGORITMOS_LAB2:
                directorio = resultados / algoritmo
                directorio.mkdir()
                (directorio / "predicciones_test.csv").write_text(
                    "fila,real,prediccion\n0,0,0\n1,1,1\n", encoding="utf-8"
                )
                (directorio / "matriz_confusion_test.csv").write_text(
                    "real,0,1\n0,1,0\n1,0,1\n", encoding="utf-8"
                )
                mejores[algoritmo] = {
                    "metadatos": {"conjunto_evaluacion": "test"},
                    "muestra_test": 2,
                    "metricas_test": {"accuracy": 1.0},
                }
            (resultados / "mejores_modelos.json").write_text(
                json.dumps(mejores), encoding="utf-8"
            )
        informe = self.raiz / "fuentes"
        informe.mkdir()
        (informe / "main.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (informe / "main.tex").write_text(
            "\\documentclass{article}\n\\begin{document}Fixture\\end{document}\n",
            encoding="utf-8",
        )
        (informe / "proyecto_overleaf.zip").write_bytes(b"fixture ZIP de fuentes")
        extra = self.raiz / "resultados" / "diabetes_reproducida"
        extra.mkdir()
        (extra / "excluir.json").write_text("{}", encoding="utf-8")
        for carpeta in (".git", ".venv", "output"):
            (self.raiz / carpeta).mkdir()
            (self.raiz / carpeta / "excluir.txt").write_text(
                "no incluir", encoding="utf-8"
            )

    def argumentos(self, salida="entrega"):
        return crear_parser().parse_args(
            [
                "--ids",
                "1",
                "2",
                "3",
                "4",
                "--informe",
                str(self.raiz / "fuentes" / "main.pdf"),
                "--fuentes-tex",
                str(self.raiz / "fuentes" / "main.tex"),
                "--resultados",
                str(self.raiz / "resultados"),
                "--salida",
                str(self.raiz / salida),
            ]
        )

    def test_entrega_repetible_manifest_verificable_y_cuatro_ids(self):
        uno = ejecutar(self.argumentos("uno"), raiz=self.raiz)
        dos = ejecutar(self.argumentos("dos"), raiz=self.raiz)
        self.assertEqual(uno.name, "1_2_3_4.zip")
        self.assertEqual(uno.read_bytes(), dos.read_bytes())
        with ZipFile(BytesIO(uno.read_bytes())) as paquete:
            nombres = set(paquete.namelist())
            self.assertIn("informe/main.pdf", nombres)
            self.assertIn("informe/main.tex", nombres)
            self.assertIn("informe/proyecto_overleaf.zip", nombres)
            self.assertIn(".streamlit/config.toml", nombres)
            self.assertIn("requirements-lab02-lock.txt", nombres)
            self.assertNotIn(b"framework_start", paquete.read("README.md"))
            self.assertIn(b"streamlit run app.py", paquete.read("README.md"))
            self.assertNotIn("salida_lab02/diabetes_reproducida/excluir.json", nombres)
            self.assertNotIn(".venv/excluir.txt", nombres)
            self.assertNotIn(".git/excluir.txt", nombres)
            manifiesto = json.loads(paquete.read("MANIFIESTO.json"))
            self.assertEqual(set(manifiesto["archivos"]), nombres - {"MANIFIESTO.json"})
            self.assertNotIn(
                str(self.raiz), paquete.read("MANIFIESTO.json").decode("utf-8")
            )
            for nombre, registro in manifiesto["archivos"].items():
                contenido = paquete.read(nombre)
                self.assertEqual(registro["bytes"], len(contenido))
                self.assertEqual(
                    registro["sha256"], hashlib.sha256(contenido).hexdigest()
                )
        self.assertEqual(ejecutar(self.argumentos("uno"), raiz=self.raiz), uno)

    def test_csv_modificado_rechaza_entrega_antes_de_crear_salida(self):
        csv = self.raiz / "data" / "lab02" / "diabetes.csv"
        csv.write_bytes(csv.read_bytes() + b"15,0\n")
        argumentos = self.argumentos()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ejecutar(argumentos, raiz=self.raiz)
        self.assertFalse(argumentos.salida.exists())

    def test_faltante_o_accuracy_incoherente_rechaza_entrega(self):
        ruta = self.raiz / "resultados" / "diabetes" / "DT" / "predicciones_test.csv"
        contenido = ruta.read_bytes()
        ruta.unlink()
        with self.assertRaisesRegex(ValueError, "13 archivos"):
            ejecutar(self.argumentos(), raiz=self.raiz)
        ruta.write_bytes(contenido.replace(b"1,1,1", b"1,1,0"))
        with self.assertRaisesRegex(ValueError, "accuracy"):
            ejecutar(self.argumentos(), raiz=self.raiz)

    def test_pdf_sin_firma_o_latex_incompleto_rechaza_entrega(self):
        argumentos = self.argumentos()
        pdf = argumentos.informe.read_bytes()
        argumentos.informe.write_text("no es un PDF", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "PDF"):
            ejecutar(argumentos, raiz=self.raiz)
        argumentos.informe.write_bytes(pdf)
        argumentos.fuentes_tex.write_text("texto incompleto", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "LaTeX completo"):
            ejecutar(argumentos, raiz=self.raiz)
        self.assertFalse(argumentos.salida.exists())


if __name__ == "__main__":
    unittest.main()
