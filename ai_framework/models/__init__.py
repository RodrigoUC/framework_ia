"""Public access point for machine learning models."""

from __future__ import annotations

import sys
from .unsupervised import Clustering, Unsupervised
from .supervised import (
    Classification,
    Regression,
    Supervised,
)

# Compatibility aliases
sys.modules.setdefault("ai_framework.models.supervisado", sys.modules.get("ai_framework.models.supervised"))
sys.modules.setdefault("ai_framework.models.no_supervisado", sys.modules.get("ai_framework.models.unsupervised"))


def __getattr__(name: str):
    if name in ("Clasificacion", "Regresion", "Supervisado", "Cluster", "NoSupervisado"):
        import ai_framework.utils.aliases as aliases
        return getattr(aliases, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # English names
    "Classification",
    "Regression",
    "Supervised",
    "Clustering",
    "Unsupervised",
    # Bilingual aliases
    "Clasificacion",
    "Regresion",
    "Supervisado",
    "Cluster",
    "NoSupervisado",
]
