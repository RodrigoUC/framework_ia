"""Comprueba el informe con evidencia sintética real, no resultados académicos."""

import ast
import hashlib
import json
import re
import shutil
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from framework_ia import Regresion
from framework_ia.datasets.concrete import FUENTE_CONCRETE, PREDICTORAS_CONCRETE
from framework_ia.evidencia import cargar_evidencia
from scripts import ejecutar_regresion
from scripts import generar_informe as informe


@pytest.fixture(scope="module")
def artefactos_informe(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("informe-sintetico")
    datos_dir = raiz / "datos"
    datos_dir.mkdir()
    rng = np.random.default_rng(23)
    datos = pd.DataFrame(
        rng.uniform(1, 40, size=(60, 8)), columns=PREDICTORAS_CONCRETE
    )
    datos["strength"] = (
        12
        + 2 * datos.cement
        - datos.water
        + 0.2 * datos.age
        + rng.normal(0, 3, len(datos))
    )
    csv = datos_dir / "concrete.csv"
    csv.write_text(datos.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    original = datos_dir / "Concrete_Data.xls"
    original.write_bytes(b"Fuente sintetica para prueba; no es el XLS academico.")
    procedencia = {
        "dataset": "uci_concrete_compressive_strength",
        "archivo": csv.name,
        "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "filas": len(datos),
        "columnas": datos.columns.tolist(),
        "fuente": FUENTE_CONCRETE,
        "lectura": {"separador": ",", "decimal": ".", "encoding": "utf-8"},
        "original": {
            "archivo": original.name,
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        },
        "derivacion": {
            "filas_originales": 80,
            "filas_preparadas": 60,
            "filas_eliminadas_duplicados": 20,
        },
    }
    (datos_dir / "procedencia.json").write_text(
        json.dumps(procedencia), encoding="utf-8"
    )
    codigo = ejecutar_regresion.main(
        [
            str(csv),
            "--target",
            "strength",
            "--feature-simple",
            "cement",
            "--salida",
            str(raiz / "resultados"),
            "--fuente",
            FUENTE_CONCRETE,
        ]
    )
    assert codigo == 0
    return raiz


@pytest.fixture
def caso_informe(artefactos_informe, tmp_path):
    destino = tmp_path / "caso"
    shutil.copytree(artefactos_informe, destino)
    return destino


def argumentos(raiz, salida=None):
    return informe.crear_parser().parse_args(
        [
            str(raiz / "resultados"),
            "--datos",
            str(raiz / "datos" / "concrete.csv"),
            "--procedencia",
            str(raiz / "datos" / "procedencia.json"),
            "--autor",
            "Estudiante de prueba (123)",
            "--fecha",
            "2026-10-21",
            "--salida",
            str(salida or raiz / "informe"),
        ]
    )


def leer_evidencia(raiz):
    return cargar_evidencia(
        raiz / "resultados",
        raiz / "datos" / "concrete.csv",
        raiz / "datos" / "procedencia.json",
    )


def archivos(raiz):
    return {
        p.relative_to(raiz).as_posix(): p.read_bytes()
        for p in raiz.rglob("*")
        if p.is_file()
    }


def generar_fixture(args):
    with patch.object(informe, "validar_preparacion_concrete") as validar_original:
        salida = informe.ejecutar(args)
    validar_original.assert_called_once()
    assert validar_original.call_args.args[0] == args.procedencia.parent
    return salida


def test_informe_refleja_catorce_candidatos_y_metricas_reales_sin_entrenar(
    caso_informe,
):
    evidencia = leer_evidencia(caso_informe)
    datos_antes = archivos(caso_informe)
    args = argumentos(caso_informe, caso_informe.parent / "informe")
    with patch.object(Regresion, "_ajustar", side_effect=AssertionError("No entrenar")):
        salida = generar_fixture(args)
    contenido = (salida / "main.tex").read_text(encoding="utf-8")
    assert "nueve modelos de regresión y catorce configuraciones" in contenido
    assert "verificó 80 filas originales y retiró 20" in contenido
    assert "conservando 60 filas" in contenido
    assert len(evidencia.comparacion) == 14
    for candidato in evidencia.comparacion.itertuples():
        etiqueta = candidato.configuracion.replace("_", r"\_")
        elegida = "Sí" if candidato.seleccionado else "No"
        fila = (
            f"{candidato.algoritmo} & {etiqueta} & {candidato.rmse_validacion:.3f} "
            f"& {candidato.puntuacion_validacion_std:.3f} & {elegida}"
        )
        assert fila in contenido
    for algoritmo, modelo in evidencia.mejores.items():
        metricas = modelo["metricas_test"]
        fila = (
            f"{algoritmo} & {modelo['metricas_train']['rmse']:.3f} "
            f"& {metricas['rmse']:.3f} & {metricas['mae']:.3f} "
            f"& {metricas['mse']:.3f} & {metricas['r2']:.3f}"
        )
        assert fila in contenido
    assert archivos(caso_informe) == datos_antes


def test_manifiesto_liga_informe_dataset_evidencia_y_seleccion(caso_informe):
    args = argumentos(caso_informe)
    salida = generar_fixture(args)
    contenido = (salida / "main.tex").read_bytes()
    manifest = json.loads(
        (salida / "manifest_informe.json").read_text(encoding="utf-8")
    )
    evidencia = leer_evidencia(caso_informe)
    assert manifest["informe_sha256"] == hashlib.sha256(contenido).hexdigest()
    assert (
        manifest["dataset_sha256"]
        == hashlib.sha256(args.datos.read_bytes()).hexdigest()
    )
    assert manifest["evidencia"] == evidencia.hashes
    assert manifest["autores"] == args.autor
    assert manifest["fecha"] == args.fecha
    assert (
        manifest["algoritmo_seleccionado"]
        == (evidencia.configuracion["experimento"]["mejor_algoritmo_por_validacion"])
    )
    assert str(caso_informe) not in json.dumps(manifest)


def test_autores_escapan_caracteres_latex_y_admiten_cuatro_integrantes(caso_informe):
    evidencia = leer_evidencia(caso_informe)
    autores = ["Ana & Luis_1 (123)", r"B%$#{x}\~^", "Persona tres", "Persona cuatro"]
    documento = informe.construir_informe(evidencia, autores, "2026-10-21")
    portada = documento.split(r"\author{", 1)[1].split(r"\date{", 1)[0]
    assert r"Ana \& Luis\_1 (123)" in portada
    assert r"B\%\$\#\{x\}\textbackslash{}\textasciitilde{}\textasciicircum{}" in portada
    assert "Persona tres" in portada and "Persona cuatro" in portada
    assert autores[0] not in portada


def test_la_seleccion_se_conserva_cuando_otro_modelo_tiene_mejor_test(caso_informe):
    evidencia = leer_evidencia(caso_informe)
    ganador = evidencia.configuracion["experimento"]["mejor_algoritmo_por_validacion"]
    alternativa = next(a for a in evidencia.mejores if a != ganador)
    mejores = deepcopy(evidencia.mejores)
    mejores[alternativa]["metricas_test"]["rmse"] = 0.0
    documento = informe.construir_informe(
        replace(evidencia, mejores=mejores), ["Estudiante"], "2026-10-21"
    )
    assert f"{informe.NOMBRES[alternativa]} obtuvo el menor RMSE de prueba" in documento
    assert (
        f"El modelo seleccionado sigue siendo {informe.NOMBRES[ganador]}" in documento
    )
    assert f"El ganador global fue {informe.NOMBRES[ganador]}" in documento


def test_repetir_mismos_argumentos_no_reescribe_artefactos(caso_informe):
    args = argumentos(caso_informe)
    salida = generar_fixture(args)
    antes = archivos(salida)
    fechas = {p.name: p.stat().st_mtime_ns for p in salida.iterdir()}
    assert generar_fixture(args) == salida
    assert archivos(salida) == antes
    assert {p.name: p.stat().st_mtime_ns for p in salida.iterdir()} == fechas


@pytest.mark.parametrize("conflicto", ["main.tex", "manifest_informe.json"])
def test_conflicto_de_salida_no_sobrescribe_ni_crea_otro_archivo(
    caso_informe, conflicto
):
    args = argumentos(caso_informe)
    args.salida.mkdir()
    (args.salida / conflicto).write_bytes(b"Contenido previo ajeno al informe.")
    (args.salida / "conservar.txt").write_bytes(b"Archivo del usuario.")
    antes = archivos(args.salida)
    with pytest.raises(ValueError, match="archivo distinto"):
        generar_fixture(args)
    assert archivos(args.salida) == antes


@pytest.mark.parametrize("alteracion", ["json", "dataset", "metricas_faltantes"])
def test_fuente_invalida_no_crea_informe(caso_informe, alteracion):
    if alteracion == "json":
        (caso_informe / "datos" / "procedencia.json").write_text("{", encoding="utf-8")
    elif alteracion == "dataset":
        ruta = caso_informe / "datos" / "concrete.csv"
        ruta.write_bytes(ruta.read_bytes() + b"\n")
    else:
        (caso_informe / "resultados" / "RLM" / "metricas.json").unlink()
    args = argumentos(caso_informe)
    antes = archivos(caso_informe)
    with (
        patch.object(informe, "validar_preparacion_concrete") as validar_original,
        pytest.raises(ValueError),
    ):
        informe.ejecutar(args)
    validar_original.assert_not_called()
    assert not args.salida.exists()
    assert archivos(caso_informe) == antes


def test_derivacion_original_invalida_bloquea_generacion(caso_informe):
    args = argumentos(caso_informe)
    with (
        patch.object(
            informe,
            "validar_preparacion_concrete",
            side_effect=ValueError("Origen distinto"),
        ),
        pytest.raises(ValueError, match="Origen distinto"),
    ):
        informe.ejecutar(args)
    assert not args.salida.exists()


@pytest.mark.parametrize("autores", [[], [" "], ["Persona", "Persona"]])
def test_autores_invalidos_no_crean_salida(caso_informe, autores):
    args = argumentos(caso_informe)
    args.autor = autores
    with pytest.raises(ValueError, match="autores"):
        generar_fixture(args)
    assert not args.salida.exists()


@pytest.mark.parametrize("fecha", ["21/10/2026", "2026-02-30", ""])
def test_fecha_invalida_no_crea_salida(caso_informe, fecha):
    args = argumentos(caso_informe)
    args.fecha = fecha
    with pytest.raises(ValueError):
        generar_fixture(args)
    assert not args.salida.exists()


def test_otro_dataset_no_se_presenta_como_concrete(caso_informe):
    evidencia = leer_evidencia(caso_informe)
    procedencia = {**evidencia.procedencia, "dataset": "otro_dataset"}
    with pytest.raises(ValueError, match="requiere Concrete"):
        informe.construir_informe(
            replace(evidencia, procedencia=procedencia), ["Estudiante"], "2026-10-21"
        )


def test_referencias_citadas_y_mayoria_de_articulos_arbitrados(caso_informe):
    documento = informe.construir_informe(
        leer_evidencia(caso_informe), ["Estudiante"], "2026-10-21"
    )
    claves = set(re.findall(r"\\bibitem\{([^}]+)\}", documento))
    citadas = {
        clave
        for grupo in re.findall(r"\\cite\{([^}]+)\}", documento)
        for clave in grupo.split(",")
    }
    articulos = {
        "yeh1998",
        "tibshirani1996",
        "hoerl1970",
        "smola2004",
        "breiman2001",
        "cawley2010",
    }
    assert citadas <= claves
    assert articulos <= citadas
    assert len(articulos) / len(claves) >= 0.5
    assert r"\begin{tikzpicture}" in documento
    assert r"\mathrm{MAE}" in documento and r"\mathrm{RMSE}" in documento
    assert r"\mathrm{MSE}" in documento and "R^2" in documento
    assert "no demuestra generalización" in documento
    assert "No se reajusta en cada pliegue interno" in documento


def test_bloque_reproduccion_python_usa_fuente_y_original_correctos(caso_informe):
    documento = informe.construir_informe(
        leer_evidencia(caso_informe), ["Estudiante"], "2026-10-21"
    )
    (bloque,) = re.findall(
        r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", documento, re.DOTALL
    )
    arbol = ast.parse(bloque)
    llamadas = {
        nodo.func.id: ast.literal_eval(nodo.args[0])
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
    }
    assert llamadas["preparar"] == ["data/lab03/concrete/Concrete_Data.xls"]
    opciones = llamadas["experimentar"]
    assert opciones[opciones.index("--fuente") + 1] == FUENTE_CONCRETE
    assert opciones[0] == "data/lab03/concrete/concrete.csv"
    assert opciones[opciones.index("--feature-simple") + 1] == "cement"
