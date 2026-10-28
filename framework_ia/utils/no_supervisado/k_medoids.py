"""Estimador de K-Medoids mediante actualización alternante de medoides."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


class ModeloKMedoids:
    """Implementa K-Medoids mediante actualización alternante de medoides.

    Expone una interfaz compatible con estimadores de scikit-learn
    (fit, fit_predict, predict, cluster_centers_, labels_, inertia_).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        metric: str = "manhattan",
        max_iter: int = 300,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def fit(self, datos) -> "ModeloKMedoids":
        """Ajusta medoides minimizando la distancia interna de cada grupo."""
        matriz = np.asarray(datos, dtype=float)
        if matriz.ndim != 2 or len(matriz) == 0:
            raise ValueError("K-Medoids requiere una matriz bidimensional no vacía.")
        if not 2 <= self.n_clusters < len(matriz):
            raise ValueError(
                "n_clusters debe ser al menos 2 y menor que el número de filas."
            )

        distancias = pairwise_distances(matriz, metric=self.metric)
        generador = np.random.default_rng(self.random_state)
        medoides = generador.choice(
            len(matriz), size=self.n_clusters, replace=False
        )

        for iteracion in range(1, self.max_iter + 1):
            dist_medoides = distancias[:, medoides]
            etiquetas = np.argmin(dist_medoides, axis=1)

            nuevos_medoides = np.empty_like(medoides)
            for k in range(self.n_clusters):
                miembros = np.where(etiquetas == k)[0]
                if len(miembros) == 0:
                    disponibles = np.setdiff1d(
                        np.arange(len(matriz)), nuevos_medoides[:k]
                    )
                    nuevos_medoides[k] = (
                        generador.choice(disponibles)
                        if len(disponibles) > 0
                        else medoides[k]
                    )
                    continue

                submatriz = distancias[np.ix_(miembros, miembros)]
                costes = submatriz.sum(axis=1)
                nuevos_medoides[k] = miembros[np.argmin(costes)]

            if np.array_equal(np.sort(medoides), np.sort(nuevos_medoides)):
                self.n_iter_ = iteracion
                break
            medoides = nuevos_medoides
        else:
            self.n_iter_ = self.max_iter

        dist_finales = distancias[:, medoides]
        self.labels_ = np.argmin(dist_finales, axis=1)
        self.medoid_indices_ = medoides
        self.cluster_centers_ = matriz[medoides]
        self.inertia_ = float(
            dist_finales[np.arange(len(matriz)), self.labels_].sum()
        )
        return self

    def fit_predict(self, datos) -> np.ndarray:
        """Ajusta el estimador y retorna las etiquetas asignadas."""
        self.fit(datos)
        assert self.labels_ is not None
        return self.labels_

    def predict(self, datos) -> np.ndarray:
        """Asigna nuevas observaciones al medoide más cercano."""
        if self.cluster_centers_ is None:
            raise RuntimeError("El modelo no está entrenado.")
        matriz = np.asarray(datos, dtype=float)
        distancias = pairwise_distances(
            matriz, self.cluster_centers_, metric=self.metric
        )
        return np.argmin(distancias, axis=1)
