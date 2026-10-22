"""Deriva Concrete Lab03 desde el XLS original local de UCI, sin acceder a la red."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework_ia.datos.concrete_lab03 import preparar_concrete

RAIZ = Path(__file__).resolve().parents[1]


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xls", type=Path, help="Concrete_Data.xls original de UCI (1030 filas).")
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "lab03" / "concrete")
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    """Comprueba todos los destinos antes de preservar el original y su derivación."""
    archivos, metadata = preparar_concrete(argumentos.xls.read_bytes())
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


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = ejecutar(argumentos)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(2, f"No se pudo preparar Concrete Lab03: {exc}\n")
    print(f"Concrete preparado; XLS original y procedencia conservados en {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
