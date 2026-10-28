"""Public access point for machine learning models."""

from __future__ import annotations

import sys
from .unsupervised import Cluster, Clustering, NoSupervisado, Unsupervised
from .supervised import (
    Clasificacion,
    Classification,
    Regresion,
    Regression,
    Supervisado,
    Supervised,
)

# Compatibility aliases
sys.modules.setdefault("ai_framework.models.supervisado", sys.modules.get("ai_framework.models.supervised"))
sys.modules.setdefault("ai_framework.models.no_supervisado", sys.modules.get("ai_framework.models.unsupervised"))

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
