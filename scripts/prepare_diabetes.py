"""Prepare Diabetes dataset for classification or materialize its regression reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_framework.datasets.diabetes import cargar_diabetes_regresion, preparar_diabetes

ROOT = Path(__file__).resolve().parents[1]
RAIZ = ROOT

__all__ = [
    "create_parser",
    "crear_parser",
    "run",
    "ejecutar",
    "main",
    "ROOT",
    "RAIZ",
    "_ejecutar_clasificacion",
    "_ejecutar_regresion",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=None,
        help="CSV original de 19 variables para clasificación; si se omite, materializa la referencia de regresión.",
    )
    parser.add_argument("--salida", "--output", dest="salida", type=Path, default=None)
    return parser


crear_parser = create_parser


def _ejecutar_clasificacion(csv_path: Path, destino: Path) -> Path:
    archivos, metadata = preparar_diabetes(csv_path.read_bytes())
    for nombre, contenido in archivos.items():
        ruta = destino / nombre
        if ruta.exists() and ruta.read_bytes() != contenido:
            raise ValueError(f"Ya existe un archivo distinto: {ruta}.")
    ruta_procedencia = destino / "procedencia.json"
    procedencia = (
        json.loads(ruta_procedencia.read_text(encoding="utf-8"))
        if ruta_procedencia.exists()
        else {}
    )
    if not isinstance(procedencia, dict):
        raise TypeError("procedencia.json debe contener un objeto JSON.")
    procedencia["diabetes"] = metadata
    serializado = (
        json.dumps(procedencia, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, contenido in archivos.items():
        (destino / nombre).write_bytes(contenido)
    ruta_procedencia.write_bytes(serializado)
    return destino


def _ejecutar_regresion(destino: Path) -> Path:
    _, contenido, metadata = cargar_diabetes_regresion()
    archivos = {
        metadata["archivo"]: contenido,
        "procedencia.json": (
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    for nombre, esperado in archivos.items():
        ruta = destino / nombre
        if ruta.exists() and (not ruta.is_file() or ruta.read_bytes() != esperado):
            raise ValueError(f"Ya existe un archivo distinto: {ruta}.")
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, esperado in archivos.items():
        ruta = destino / nombre
        if not ruta.exists():
            ruta.write_bytes(esperado)
    return destino


def run(argumentos: argparse.Namespace) -> Path:
    """Valida todos los destinos antes de escribir los archivos de preparación."""
    csv_path = getattr(argumentos, "csv", None)
    salida = getattr(argumentos, "salida", None)
    if csv_path is not None:
        return _ejecutar_clasificacion(csv_path, salida or (ROOT / "data" / "lab02"))
    return _ejecutar_regresion(salida or (ROOT / "data" / "lab03"))


ejecutar = run


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = run(argumentos)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"No se pudo preparar Diabetes: {exc}\n")
    print(f"Diabetes preparado y procedencia registrada en {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
