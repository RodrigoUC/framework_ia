"""Prepara Potabilidad: casos completos del original, sin imputar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework_ia.datos.potabilidad import preparar_potabilidad

RAIZ = Path(__file__).resolve().parents[1]


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", type=Path, help="water_potability.csv original (3276 filas)."
    )
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "lab02")
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    archivos, metadata = preparar_potabilidad(argumentos.csv.read_bytes())
    destino = argumentos.salida
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
    procedencia["potabilidad"] = metadata
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, contenido in archivos.items():
        (destino / nombre).write_bytes(contenido)
    ruta_procedencia.write_bytes(
        (
            json.dumps(procedencia, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    )
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = ejecutar(argumentos)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"No se pudo preparar Potabilidad: {exc}\n")
    print(f"Potabilidad preparada y procedencia registrada en {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
