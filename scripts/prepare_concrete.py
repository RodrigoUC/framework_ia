"""Derive Concrete dataset from original local UCI XLS file without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_framework.datasets.concrete import preparar_concrete as prepare_concrete

ROOT = Path(__file__).resolve().parents[1]
RAIZ = ROOT

__all__ = ["create_parser", "crear_parser", "run", "ejecutar", "main"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "xls", type=Path, help="Original UCI Concrete_Data.xls (1030 rows)."
    )
    parser.add_argument(
        "--salida", "--output", dest="salida", type=Path, default=ROOT / "data" / "lab03" / "concrete"
    )
    return parser


crear_parser = create_parser


def run(argumentos: argparse.Namespace) -> Path:
    """Check all destinations before preserving original and derivation."""
    archivos, metadata = prepare_concrete(argumentos.xls.read_bytes())
    archivos["procedencia.json"] = (
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    destino = argumentos.salida
    if destino.exists() and not destino.is_dir():
        raise ValueError(f"La salida debe ser un directorio: {destino}.")
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


ejecutar = run


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = run(argumentos)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"No se pudo preparar Concrete: {exc}\n")
    print(f"Concrete preparado; XLS original y procedencia conservados en {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
