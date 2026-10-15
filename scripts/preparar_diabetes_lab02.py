"""Prepara Diabetes de Lab02 sin imputar ni escalar las predictoras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework_ia.datos.diabetes_lab02 import preparar_diabetes

RAIZ = Path(__file__).resolve().parents[1]


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV original de 19 variables.")
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "lab02")
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    archivos, metadata = preparar_diabetes(argumentos.csv.read_bytes())
    destino = argumentos.salida
    for nombre, contenido in archivos.items():
        ruta = destino / nombre
        if ruta.exists() and ruta.read_bytes() != contenido:
            raise ValueError(f"Ya existe un archivo distinto: {ruta}.")
    ruta_procedencia = destino / "procedencia.json"
    procedencia = (
        json.loads(ruta_procedencia.read_text(encoding="utf-8"))
        if ruta_procedencia.exists() else {}
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


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        destino = ejecutar(argumentos)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"No se pudo preparar Diabetes: {exc}\n")
    print(f"Diabetes preparado y procedencia registrada en {destino}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
