"""Compatibility wrapper for prepare_potability.py."""
from __future__ import annotations

from scripts.prepare_potability import (
    RAIZ,
    ROOT,
    crear_parser,
    create_parser,
    ejecutar,
    main,
    run,
)

__all__ = ["RAIZ", "ROOT", "crear_parser", "create_parser", "ejecutar", "run", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
