"""Compatibility wrapper for run_agents.py."""
from __future__ import annotations

from scripts.run_agents import crear_parser, create_parser, ejecutar, main, run

__all__ = ["crear_parser", "create_parser", "ejecutar", "main", "run"]

if __name__ == "__main__":
    raise SystemExit(main())
