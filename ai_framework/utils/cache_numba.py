"""Numba cache directory configuration (required by UMAP)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_numba_cache() -> Path:
    """Return a writable directory for Numba cache.

    Some Streamlit sessions inherit ``NUMBA_CACHE_DIR`` from the system.
    If that path cannot be created or written to, keeping it prevents importing
    UMAP even if a temporary directory is available.
    """
    configurado = os.environ.get("NUMBA_CACHE_DIR")
    if configurado:
        try:
            directorio = Path(configurado).expanduser()
            directorio.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directorio):
                pass
            return directorio
        except (OSError, ValueError):
            pass

    directorio = Path(tempfile.gettempdir()) / "framework_start_numba"
    directorio.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(directorio)
    return directorio


configurar_cache_numba = configure_numba_cache
