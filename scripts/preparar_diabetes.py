"""Compatibility wrapper for prepare_diabetes.py."""
from __future__ import annotations

from scripts.prepare_diabetes import (
    RAIZ,
    ROOT,
    _ejecutar_clasificacion,
    _ejecutar_regresion,
    crear_parser,
    create_parser,
    ejecutar,
    main,
    run,
)

__all__ = [
    "RAIZ",
    "ROOT",
    "_ejecutar_clasificacion",
    "_ejecutar_regresion",
    "crear_parser",
    "create_parser",
    "ejecutar",
    "run",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
