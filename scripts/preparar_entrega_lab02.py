"""Empaqueta únicamente código, datos y evidencia verificada de LAB02."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from framework_ia.exportacion_lab02 import construir_zip_experimento
from framework_ia.utils import ALGORITMOS_LAB2

RAIZ = Path(__file__).resolve().parents[1]
DATASETS = ("potabilidad", "diabetes")
EXCLUIDOS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida y crea el ZIP de LAB02.")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument(
        "--informe", type=Path, required=True, help="Informe PDF final."
    )
    parser.add_argument(
        "--fuentes-tex",
        "--fuentesTEX",
        dest="fuentes_tex",
        type=Path,
        required=True,
        help="TEX principal autónomo del informe.",
    )
    parser.add_argument("--resultados", type=Path, default=RAIZ / "salida_lab02")
    parser.add_argument("--salida", type=Path, default=RAIZ / "output" / "lab02")
    return parser


def _sha(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def _json(ruta: Path) -> dict:
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        raise TypeError(f"Se esperaba un objeto JSON: {ruta.name}.")
    return contenido


def _validar_resultados(resultados: Path, datos: Path) -> dict[str, str]:
    """Rechaza ejecuciones parciales o evidencia de otro CSV."""
    huellas = {}
    algoritmos = set(ALGORITMOS_LAB2)
    esperados = {
        "configuracion.json",
        "comparacion_validacion.csv",
        "mejores_modelos.json",
    }
    esperados.update(
        f"{algoritmo}/{nombre}"
        for algoritmo in algoritmos
        for nombre in ("matriz_confusion_test.csv", "predicciones_test.csv")
    )
    for nombre in DATASETS:
        directorio = resultados / nombre
        presentes = {
            ruta.relative_to(directorio).as_posix()
            for ruta in directorio.rglob("*")
            if ruta.is_file()
        }
        if presentes != esperados:
            raise ValueError(
                f"{nombre}: se requieren exactamente los 13 archivos de evidencia."
            )
        configuracion = _json(directorio / "configuracion.json")
        dataset = configuracion["dataset"]
        archivo = dataset["archivo"]
        if not isinstance(archivo, str) or Path(archivo).name != archivo:
            raise ValueError(f"{nombre}: nombre del CSV inválido.")
        csv = datos / archivo
        huella = _sha(csv.read_bytes())
        if dataset.get("sha256") != huella:
            raise ValueError(
                f"{nombre}: SHA-256 de resultados no coincide con el CSV original."
            )
        metadatos = configuracion["experimento"]
        if (
            metadatos.get("candidatos_fallidos") != 0
            or metadatos.get("candidatos_exitosos") != 15
            or metadatos.get("errores_prueba")
            or metadatos.get("test_usado_para_seleccion") is not False
        ):
            raise ValueError(
                f"{nombre}: experimento incompleto o selección sin test reservado."
            )
        tabla = pd.read_csv(directorio / "comparacion_validacion.csv")
        if (
            len(tabla) != 15
            or set(tabla.algoritmo) != algoritmos
            or not tabla.estado.eq("ok").all()
            or not tabla.groupby("algoritmo").size().eq(3).all()
            or not tabla.loc[tabla.configuracion.eq("estandar")]
            .groupby("algoritmo")
            .size()
            .reindex(ALGORITMOS_LAB2, fill_value=0)
            .eq(1)
            .all()
        ):
            raise ValueError(
                f"{nombre}: faltan configuraciones estándar y dos variantes por método."
            )
        seleccionadas = tabla.loc[tabla.seleccionado.astype(str).str.lower().eq("true")]
        if len(seleccionadas) != 5 or set(seleccionadas.algoritmo) != algoritmos:
            raise ValueError(f"{nombre}: debe existir un ganador por método.")
        mejores = _json(directorio / "mejores_modelos.json")
        if set(mejores) != algoritmos:
            raise ValueError(f"{nombre}: faltan ganadores evaluados en test.")
        for algoritmo, ganador in mejores.items():
            if ganador["metadatos"].get("conjunto_evaluacion") != "test":
                raise ValueError(
                    f"{nombre}/{algoritmo}: evaluación final no identificada como test."
                )
            predicciones = pd.read_csv(directorio / algoritmo / "predicciones_test.csv")
            if (
                predicciones.empty
                or list(predicciones.columns) != ["fila", "real", "prediccion"]
                or len(predicciones) != ganador["muestra_test"]
            ):
                raise ValueError(f"{nombre}/{algoritmo}: predicciones incompletas.")
            observada = float(predicciones.real.eq(predicciones.prediccion).mean())
            accuracy = ganador["metricas_test"]["accuracy"]
            if not math.isfinite(accuracy) or abs(observada - accuracy) > 1e-12:
                raise ValueError(
                    f"{nombre}/{algoritmo}: accuracy no coincide con predicciones."
                )
            matriz = pd.read_csv(
                directorio / algoritmo / "matriz_confusion_test.csv", index_col=0
            )
            valores = matriz.apply(pd.to_numeric, errors="raise")
            if (
                valores.empty
                or valores.isna().any().any()
                or (valores < 0).any().any()
                or not valores.eq(valores.round()).all().all()
                or int(valores.to_numpy().sum()) != len(predicciones)
            ):
                raise ValueError(
                    f"{nombre}/{algoritmo}: matriz de confusión incompleta."
                )
        huellas[nombre] = huella
    if len(set(huellas.values())) != 2:
        raise ValueError("Los dos experimentos deben corresponder a CSV distintos.")
    return huellas


def construir_entrega(
    argumentos: argparse.Namespace, *, raiz: Path = RAIZ
) -> dict[str, bytes]:
    """Valida todo antes de crear archivos de salida; genera un manifiesto relativo."""
    if (
        not argumentos.ids
        or any(not identificador.isdecimal() for identificador in argumentos.ids)
        or len(set(argumentos.ids)) != len(argumentos.ids)
    ):
        raise ValueError("Indique IDs numéricos no repetidos.")
    pdf = argumentos.informe.read_bytes()
    if not pdf.startswith(b"%PDF-") or len(pdf) < 10:
        raise ValueError("--informe debe contener un PDF real.")
    tex = argumentos.fuentes_tex.read_text(encoding="utf-8")
    if not all(
        marca in tex
        for marca in ("\\documentclass", "\\begin{document}", "\\end{document}")
    ):
        raise ValueError("--fuentes-tex debe contener un documento LaTeX completo.")
    for carpeta in ("framework_ia", "scripts", "tests"):
        if not any(ruta.is_file() for ruta in (raiz / carpeta).rglob("*.py")):
            raise ValueError(f"Falta el código Python requerido: {carpeta}.")
    datos = raiz / "data" / "lab02"
    _json(datos / "procedencia.json")
    huellas = _validar_resultados(argumentos.resultados, datos)
    archivos = {}

    def agregar(ruta: Path, nombre: str) -> None:
        if any(parte in EXCLUIDOS for parte in ruta.parts):
            return
        contenido = ruta.read_bytes()
        if nombre in archivos and archivos[nombre] != contenido:
            raise ValueError(
                f"Dos archivos distintos tienen el mismo destino: {nombre}."
            )
        archivos[nombre] = contenido

    for nombre in (
        "app.py",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-lab02-lock.txt",
        ".streamlit/config.toml",
    ):
        agregar(raiz / nombre, nombre)
    leeme = raiz / "docs" / "lab02" / "LEEME_ENTREGA.md"
    agregar(leeme, "README.md")
    for carpeta in ("framework_ia", "scripts", "tests"):
        for ruta in sorted((raiz / carpeta).rglob("*.py")):
            if ruta.name in {"ejecutar_lab01.py", "preparar_entrega_lab01.py"}:
                continue
            agregar(ruta, ruta.relative_to(raiz).as_posix())
    for ruta in sorted((raiz / "docs" / "lab02").rglob("*")):
        if ruta.is_file() and ruta.suffix.lower() in {".md", ".tex", ".bib", ".pdf"}:
            agregar(ruta, ruta.relative_to(raiz).as_posix())
    guia = raiz / "docs" / "lab01" / "guia_clasificacion.pdf"
    if guia.is_file():
        # Es la guía LAB02; conservar su ruta evita romper el enlace del README.
        agregar(guia, "docs/lab01/guia_clasificacion.pdf")
    nombres_datos = {"procedencia.json", "README.md"}
    nombres_datos.update(
        _json(argumentos.resultados / nombre / "configuracion.json")["dataset"][
            "archivo"
        ]
        for nombre in DATASETS
    )
    for nombre in sorted(nombres_datos):
        ruta = datos / nombre
        if ruta.is_file():
            agregar(ruta, ruta.relative_to(raiz).as_posix())
    for nombre in DATASETS:
        for ruta in sorted((argumentos.resultados / nombre).rglob("*")):
            if ruta.is_file():
                agregar(
                    ruta,
                    "salida_lab02/"
                    + ruta.relative_to(argumentos.resultados).as_posix(),
                )
    agregar(argumentos.informe, "informe/main.pdf")
    for ruta in sorted(argumentos.fuentes_tex.parent.rglob("*")):
        if ruta.is_file() and ruta.suffix.lower() in {
            ".tex",
            ".bib",
            ".sty",
            ".cls",
            ".bst",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".csv",
            ".json",
            ".md",
            ".zip",
        }:
            agregar(
                ruta,
                "informe/" + ruta.relative_to(argumentos.fuentes_tex.parent).as_posix(),
            )
    manifiesto = {
        "laboratorio": "Lab02",
        "integrantes": argumentos.ids,
        "sha256_datasets": huellas,
        "archivos": {
            nombre: {"sha256": _sha(contenido), "bytes": len(contenido)}
            for nombre, contenido in sorted(archivos.items())
        },
        "nota": "El manifiesto registra todos los archivos salvo el propio manifiesto.",
    }
    archivos["MANIFIESTO.json"] = (
        json.dumps(manifiesto, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    return archivos


def ejecutar(argumentos: argparse.Namespace, *, raiz: Path = RAIZ) -> Path:
    archivos = construir_entrega(argumentos, raiz=raiz)
    contenido = construir_zip_experimento(archivos)
    destino = argumentos.salida / ("_".join(argumentos.ids) + ".zip")
    if destino.exists() and destino.read_bytes() != contenido:
        raise ValueError("Ya existe una entrega distinta; elija otra carpeta --salida.")
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(contenido)
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = ejecutar(argumentos)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"No se pudo empaquetar LAB02: {exc}\n")
    print(f"Entrega verificada: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
