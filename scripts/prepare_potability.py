"""Prepare Potability: complete cases from original data without imputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_framework.datasets.potability import preparar_potabilidad as prepare_potability

ROOT = Path(__file__).resolve().parents[1]
RAIZ = ROOT

__all__ = ["create_parser", "crear_parser", "run", "ejecutar", "main"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", type=Path, help="Original water_potability.csv (3276 rows)."
    )
    parser.add_argument("--salida", "--output", dest="salida", type=Path, default=ROOT / "data" / "lab02")
    return parser


crear_parser = create_parser


def run(argumentos: argparse.Namespace) -> Path:
    archivos, metadata = prepare_potability(argumentos.csv.read_bytes())
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


ejecutar = run


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = run(argumentos)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"No se pudo preparar Potabilidad: {exc}\n")
    print(f"Potability prepared and provenance recorded in {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
