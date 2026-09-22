"""Empaqueta el código, informe y resultados bajo el nombre exigido por Lab01."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


RAIZ = Path(__file__).resolve().parents[1]
EXCLUIDOS = {"__pycache__", ".git", ".pytest_cache", "salida_lab01"}


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea el ZIP de entrega de Lab01.")
    parser.add_argument("--ids", nargs="+", required=True, help="Uno o dos ID de integrantes.")
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RAIZ / "salida_lab01",
        help="Directorio generado por scripts.ejecutar_lab01.",
    )
    parser.add_argument(
        "--informe",
        type=Path,
        required=True,
        help="PDF final generado con la plantilla oficial de Overleaf.",
    )
    return parser.parse_args()


def _agregar(archivo_zip: ZipFile, ruta: Path) -> None:
    """Agrega una ruta relativa a la raíz, sin cachés ni repositorio Git."""
    if any(parte in EXCLUIDOS for parte in ruta.parts):
        return
    archivo_zip.write(ruta, ruta.relative_to(RAIZ))


def ejecutar() -> None:
    args = _argumentos()
    if not 1 <= len(args.ids) <= 2 or any(not valor.strip() for valor in args.ids):
        raise ValueError("Indique uno o dos ID no vacíos.")
    informe = args.informe.resolve()
    resultados = args.resultados.resolve()
    if not informe.is_file() or informe.suffix.lower() != ".pdf":
        raise ValueError("--informe debe señalar el PDF final del laboratorio.")
    if not resultados.is_dir():
        raise ValueError("Ejecute primero scripts.ejecutar_lab01 para crear los resultados.")

    destino = RAIZ / f"{'_'.join(args.ids)}.zip"
    with ZipFile(destino, "w", compression=ZIP_DEFLATED) as archivo_zip:
        for ruta in RAIZ.rglob("*"):
            if ruta.is_file() and ruta != destino:
                _agregar(archivo_zip, ruta)
        for ruta in resultados.rglob("*"):
            if ruta.is_file():
                archivo_zip.write(ruta, Path("resultados") / ruta.relative_to(resultados))
        archivo_zip.write(informe, Path("informe") / informe.name)
    print(f"Entrega creada: {destino}")


if __name__ == "__main__":
    ejecutar()
