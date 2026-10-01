"""Configuración del directorio de caché de Numba (requerido por UMAP)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configurar_cache_numba() -> Path:
    """Devuelve un directorio escribible para la caché de Numba.

    Algunas sesiones de Streamlit heredan ``NUMBA_CACHE_DIR`` desde el
    sistema. Si esa ruta no se puede crear o escribir, conservarla impide
    importar UMAP aunque exista una carpeta temporal disponible.
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
