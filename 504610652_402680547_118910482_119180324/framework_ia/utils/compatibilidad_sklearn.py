"""Constructores de objetos sklearn compatibles con distintas versiones."""

from __future__ import annotations

from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder


def crear_codificador_denso() -> OneHotEncoder:
    """Crea un codificador one-hot denso compatible con distintas versiones."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def crear_tsne(
    *, perplexity: float, max_iter: int, random_state: int
) -> TSNE:
    """Crea un TSNE bidimensional; scikit-learn >= 1.5 renombró n_iter a max_iter."""
    parametros = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
    }
    try:
        return TSNE(max_iter=max_iter, **parametros)
    except TypeError as exc:
        if "max_iter" not in str(exc):
            raise
        return TSNE(n_iter=max_iter, **parametros)
