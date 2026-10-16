"""Materializa la referencia local de regresión sin descargar, imputar ni escalar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework_ia.datos.diabetes_lab03 import cargar_diabetes_regresion

RAIZ = Path(__file__).resolve().parents[1]


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "lab03")
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    """Valida todos los destinos antes de escribir; repetir los mismos bytes es seguro."""
    _, contenido, metadata = cargar_diabetes_regresion()
    archivos = {
        metadata["archivo"]: contenido,
        "procedencia.json": (
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    destino = argumentos.salida
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


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = ejecutar(argumentos)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"No se pudo preparar la referencia Lab03: {exc}\n")
    print(f"Referencia de regresión y procedencia registradas en {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
