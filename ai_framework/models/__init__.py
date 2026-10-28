"""AI Framework Models module."""

from .supervised import (
    Clasificacion,
    Classification,
    Regresion,
    Regression,
    Supervisado,
    Supervised,
)
from .unsupervised import (
    Cluster,
    Clustering,
    NoSupervisado,
    Unsupervised,
)

__all__ = [
    "Classification",
    "Regression",
    "Supervised",
    "Clustering",
    "Unsupervised",
    "Clasificacion",
    "Regresion",
    "Supervisado",
    "Cluster",
    "NoSupervisado",
]
