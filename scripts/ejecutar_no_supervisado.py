"""Compatibility wrapper for run_unsupervised.py."""
from __future__ import annotations

from scripts.run_unsupervised import (
    _argumentos,
    _confiabilidad,
    _crear_cluster,
    _crear_reductor,
    _guardar_json,
    _json,
    ejecutar,
    run,
)

__all__ = [
    "_argumentos",
    "_confiabilidad",
    "_crear_cluster",
    "_crear_reductor",
    "_guardar_json",
    "_json",
    "ejecutar",
    "run",
]

if __name__ == "__main__":
    run()
