"""Empaqueta únicamente código, datos y evidencia verificada de LAB02."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from framework_ia.datos.diabetes_lab02 import (
    PREDICTORAS_DIABETES,
    validar_preparacion_diabetes,
)
from framework_ia.datos.potabilidad_lab02 import validar_preparacion_potabilidad
from framework_ia.exportacion import construir_zip_experimento
from framework_ia.utils import ALGORITMOS_LAB2, configuraciones_experimento

RAIZ = Path(__file__).resolve().parents[1]
DATASETS = ("potabilidad", "diabetes")
EXCLUIDOS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache"}
CON_IMPORTANCIA_INTERNA = {"DT", "RF", "XGBoost", "AdaBoost"}


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


def archivos_evidencia() -> tuple[set[str], set[str]]:
    """Archivos obligatorios y opcionales de cada carpeta de resultados."""
    obligatorios = {
        "configuracion.json",
        "comparacion_validacion.csv",
        "mejores_modelos.json",
        "ranking_atributos.csv",
        "DT/arbol.txt",
    }
    for algoritmo in ALGORITMOS_LAB2:
        obligatorios.update(
            f"{algoritmo}/{nombre}"
            for nombre in (
                "matriz_confusion_test.csv",
                "predicciones_test.csv",
                "importancia_permutacion.csv",
                "roc_test.csv",
            )
        )
    opcionales = {f"{a}/importancia_modelo.csv" for a in CON_IMPORTANCIA_INTERNA}
    return obligatorios, opcionales


def _validar_resultados(
    resultados: Path, datos: Path, procedencia: dict
) -> dict[str, str]:
    """Rechaza ejecuciones parciales, otro protocolo o evidencia de otro CSV."""
    huellas = {}
    algoritmos = set(ALGORITMOS_LAB2)
    catalogo = configuraciones_experimento()
    total = sum(len(lista) for lista in catalogo.values())
    obligatorios, opcionales = archivos_evidencia()
    for nombre in DATASETS:
        directorio = resultados / nombre
        presentes = {
            ruta.relative_to(directorio).as_posix()
            for ruta in directorio.rglob("*")
            if ruta.is_file()
        }
        if not obligatorios <= presentes or not presentes <= obligatorios | opcionales:
            faltan = sorted(obligatorios - presentes)
            sobran = sorted(presentes - obligatorios - opcionales)
            raise ValueError(
                f"{nombre}: evidencia incompleta o ajena (faltan {faltan}; sobran {sobran})."
            )
        configuracion = _json(directorio / "configuracion.json")
        dataset = configuracion["dataset"]
        archivo = dataset["archivo"]
        if not isinstance(archivo, str) or Path(archivo).name != archivo:
            raise ValueError(f"{nombre}: nombre del CSV inválido.")
        origen = procedencia[nombre]
        if archivo != origen["ruta"]:
            raise ValueError(f"{nombre}: resultados de otro archivo de entrada.")
        huella = _sha((datos / archivo).read_bytes())
        if dataset.get("sha256") != huella or origen.get("sha256") != huella:
            raise ValueError(
                f"{nombre}: SHA-256 de resultados y procedencia debe coincidir con el CSV de entrada."
            )
        if dataset.get("fuente") != origen["fuente"]:
            raise ValueError(f"{nombre}: fuente de resultados distinta de procedencia.")
        if dataset.get("filas") != origen["derivacion"]["filas_preparadas"]:
            raise ValueError(f"{nombre}: filas distintas de la preparación registrada.")
        argumentos = configuracion["argumentos"]
        if nombre == "diabetes" and (
            argumentos.get("target") != "diabetes"
            or argumentos.get("features") != list(PREDICTORAS_DIABETES)
            or argumentos.get("incluir_categoricas") is not True
            or dataset.get("columnas") != [*PREDICTORAS_DIABETES, "diabetes"]
        ):
            raise ValueError(
                "diabetes: resultados incompatibles con el protocolo de preparación y sus 17 predictoras."
            )
        if nombre == "potabilidad" and argumentos.get("target") != "Potability":
            raise ValueError("potabilidad: el objetivo debe ser Potability.")
        metadatos = configuracion["experimento"]
        if (
            metadatos.get("estrategia_seleccion") != "validacion_cruzada_estratificada"
            or metadatos.get("candidatos_fallidos") != 0
            or metadatos.get("candidatos_exitosos") != total
            or metadatos.get("errores_prueba")
            or metadatos.get("test_usado_para_seleccion") is not False
        ):
            raise ValueError(
                f"{nombre}: experimento incompleto, sin validación cruzada o sin test reservado."
            )
        tabla = pd.read_csv(directorio / "comparacion_validacion.csv")
        configs = tabla[tabla.etapa == "configuracion"]
        if (
            len(configs) != total
            or set(configs.algoritmo) != algoritmos
            or not tabla.estado.eq("ok").all()
            or any(
                (configs.algoritmo == a).sum() != len(catalogo[a]) for a in algoritmos
            )
            or any(
                ((configs.algoritmo == a) & (configs.configuracion == "estandar")).sum() != 1
                for a in algoritmos
            )
        ):
            raise ValueError(
                f"{nombre}: faltan configuraciones estándar y variantes por método."
            )
        seleccionadas = tabla.loc[tabla.seleccionado.astype(str).str.lower().eq("true")]
        if len(seleccionadas) != 5 or set(seleccionadas.algoritmo) != algoritmos:
            raise ValueError(f"{nombre}: debe existir un ganador por método.")
        for algoritmo, grupo in tabla.groupby("algoritmo"):
            elegido = grupo.loc[grupo.seleccionado.astype(str).str.lower().eq("true")]
            if not math.isclose(
                float(elegido.puntuacion_validacion.iloc[0]),
                float(grupo.puntuacion_validacion.max()),
                abs_tol=1e-12,
            ):
                raise ValueError(f"{nombre}/{algoritmo}: el ganador no maximiza la validación.")
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
    procedencia = _json(datos / "procedencia.json")
    archivos_datos = {
        **validar_preparacion_diabetes(datos, procedencia["diabetes"]),
        **validar_preparacion_potabilidad(datos, procedencia["potabilidad"]),
    }
    huellas = _validar_resultados(argumentos.resultados, datos, procedencia)
    archivos = {}

    def agregar(ruta: Path, nombre: str) -> None:
        partes = ruta.relative_to(raiz).parts if ruta.is_relative_to(raiz) else ruta.parts
        if any(parte in EXCLUIDOS for parte in partes):
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


        ".streamlit/config.toml",
    ):
        agregar(raiz / nombre, nombre)
    agregar(raiz / "docs" / "lab02" / "LEEME_ENTREGA.md", "README.md")
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
        agregar(guia, "docs/lab01/guia_clasificacion.pdf")
    ejemplo = raiz / "data" / "ejemplo_analisis.csv"
    if ejemplo.is_file():  # Dataset de ejemplo que la aplicación abre por defecto.
        agregar(ejemplo, "data/ejemplo_analisis.csv")
    for nombre, contenido in archivos_datos.items():
        archivos[f"data/lab02/{nombre}"] = contenido
    for nombre in ("procedencia.json", "README.md", "trazabilidad_diabetes.csv"):
        ruta = datos / nombre
        if ruta.is_file():
            agregar(ruta, ruta.relative_to(raiz).as_posix())
    for nombre in DATASETS:
        for ruta in sorted((argumentos.resultados / nombre).rglob("*")):
            if ruta.is_file():
                archivos[
                    "salida_lab02/" + ruta.relative_to(argumentos.resultados).as_posix()
                ] = ruta.read_bytes()
    archivos["informe/main.pdf"] = pdf
    for ruta in sorted(argumentos.fuentes_tex.parent.rglob("*")):
        if ruta.is_file() and ruta.suffix.lower() in {
            ".tex", ".bib", ".sty", ".cls", ".bst", ".png", ".jpg", ".jpeg",
            ".svg", ".csv", ".json", ".md", ".zip",
        }:
            archivos[
                "informe/" + ruta.relative_to(argumentos.fuentes_tex.parent).as_posix()
            ] = ruta.read_bytes()
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
