"""Sklearn object constructors compatible across multiple versions."""

from __future__ import annotations

from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder


def create_dense_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible across different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_tsne(
    *, perplexity: float, max_iter: int, random_state: int
) -> TSNE:
    """Create a 2D TSNE; scikit-learn >= 1.5 renamed n_iter to max_iter."""
    params = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
    }
    try:
        return TSNE(max_iter=max_iter, **params)
    except TypeError as exc:
        if "max_iter" not in str(exc):
            raise
        return TSNE(n_iter=max_iter, **params)


# Backward compatibility aliases
crear_codificador_denso = create_dense_encoder
crear_tsne = create_tsne
