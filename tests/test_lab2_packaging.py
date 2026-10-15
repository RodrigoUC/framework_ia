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

from framework_ia.datos.diabetes_lab02 import (
    PREDICTORAS_DIABETES,
    preparar_diabetes,
)
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
        self.datos = self.raiz / "data" / "lab02"
        self.datos.mkdir(parents=True)
        columnas_originales = [
            "id",
            "chol",
            "stab.glu",
            "hdl",
            "ratio",
            "glyhb",
            "location",
            "age",
            "gender",
            "height",
            "weight",
            "frame",
            "bp.1s",
            "bp.1d",
            "bp.2s",
            "bp.2d",
            "waist",
            "hip",
            "time.ppn",
        ]
        original = pd.DataFrame(
            {columna: range(10, 18) for columna in columnas_originales}
        )
        original["id"] = range(1000, 1008)
        original["glyhb"] = [4.31, 8.0, None, 7.0, 7.01, 5.0, None, 6.0]
        original["location"] = ["Buckingham", "Louisa"] * 4
        original["gender"] = ["female", "male"] * 4
        original["frame"] = ["medium", "large"] * 4
        self.archivos_diabetes, metadata = preparar_diabetes(
            original.to_csv(index=False, lineterminator="\n").encode("utf-8")
        )
        for nombre, contenido in self.archivos_diabetes.items():
            (self.datos / nombre).write_bytes(contenido)
        potabilidad = self.datos / "potabilidad.csv"
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_csv(
            potabilidad, index=False
        )
        self.procedencia = {
            "potabilidad": {
                "ruta": potabilidad.name,
                "sha256": hashlib.sha256(potabilidad.read_bytes()).hexdigest(),
                "fuente": "fixture sintético de potabilidad",
            },
            "diabetes": metadata,
        }
        (self.datos / "procedencia.json").write_text(
            json.dumps(self.procedencia), encoding="utf-8"
        )
        for nombre in ("potabilidad", "diabetes"):
            origen = self.procedencia[nombre]
            csv = self.datos / origen["ruta"]
            tabla_csv = pd.read_csv(csv)
            argumentos = {
                "target": "diabetes" if nombre == "diabetes" else "target",
                "features": list(PREDICTORAS_DIABETES)
                if nombre == "diabetes"
                else ["x"],
                "incluir_categoricas": nombre == "diabetes",
                "separador": ",",
                "decimal": ".",
                "encoding": "utf-8",
            }
            resultados = self.raiz / "resultados" / nombre
            resultados.mkdir(parents=True)
            configuracion = {
                "argumentos": argumentos,
                "dataset": {
                    "archivo": csv.name,
                    "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                    "fuente": origen["fuente"],
                    "filas": len(tabla_csv),
                    "columnas": list(tabla_csv.columns),
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
            csv_datos = {
                nombre
                for nombre in nombres
                if nombre.startswith("data/lab02/") and nombre.endswith(".csv")
            }
            self.assertEqual(
                csv_datos,
                {
                    "data/lab02/potabilidad.csv",
                    "data/lab02/diabetes_original.csv",
                    "data/lab02/diabetes.csv",
                    "data/lab02/trazabilidad_diabetes.csv",
                },
            )
            for nombre, contenido in self.archivos_diabetes.items():
                self.assertEqual(paquete.read(f"data/lab02/{nombre}"), contenido)
            for nombre in ("potabilidad", "diabetes"):
                self.assertEqual(
                    sum(ruta.startswith(f"salida_lab02/{nombre}/") for ruta in nombres),
                    13,
                )
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
        with self.assertRaises(ValueError):
            ejecutar(argumentos, raiz=self.raiz)
        self.assertFalse(argumentos.salida.exists())

    def test_original_trazabilidad_o_regla_modificados_rechazan_entrega(self):
        alteraciones = {
            "diabetes_original.csv": lambda contenido: contenido.replace(
                b"4.31", b"4.32"
            ),
            "trazabilidad_diabetes.csv": lambda contenido: contenido + b"fila extra\n",
            "procedencia.json": lambda contenido: contenido.replace(
                b'"umbral": 7.0', b'"umbral": 8.0'
            ),
        }
        for nombre, alterar in alteraciones.items():
            with self.subTest(nombre=nombre):
                ruta = self.datos / nombre
                previo = ruta.read_bytes()
                cambiado = alterar(previo)
                self.assertNotEqual(cambiado, previo)
                ruta.write_bytes(cambiado)
                try:
                    argumentos = self.argumentos()
                    with self.assertRaises(ValueError):
                        ejecutar(argumentos, raiz=self.raiz)
                    self.assertFalse(argumentos.salida.exists())
                finally:
                    ruta.write_bytes(previo)

    def test_resultados_anteriores_o_predictora_del_target_rechazan_entrega(self):
        ruta = self.raiz / "resultados" / "diabetes" / "configuracion.json"
        previo = ruta.read_bytes()
        configuracion = json.loads(previo)
        configuracion["dataset"]["sha256"] = "0" * 64
        ruta.write_text(json.dumps(configuracion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ejecutar(self.argumentos(), raiz=self.raiz)
        configuracion = json.loads(previo)
        configuracion["argumentos"]["features"].append("glyhb")
        ruta.write_text(json.dumps(configuracion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "17 predictoras"):
            ejecutar(self.argumentos(), raiz=self.raiz)
        self.assertFalse(self.argumentos().salida.exists())

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
