"""Implementaciones de agrupamiento no provistas por sklearn base."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


class ModeloKMedoids:
    """Implementa K-Medoids mediante actualización alternante de medoides.

    La clase expone una interfaz compatible con los atributos principales de
    sklearn para que el orquestador no dependa de una biblioteca adicional.
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
            etiquetas = np.argmin(distancias[:, medoides], axis=1)
            nuevos = medoides.copy()
            for cluster in range(self.n_clusters):
                miembros = np.flatnonzero(etiquetas == cluster)
                if len(miembros) == 0:
                    nuevos[cluster] = self._seleccionar_reemplazo(
                        distancias, nuevos, generador
                    )
                    continue
                internas = distancias[np.ix_(miembros, miembros)]
                nuevos[cluster] = miembros[np.argmin(internas.sum(axis=1))]

            self.n_iter_ = iteracion
            if np.array_equal(nuevos, medoides):
                break
            medoides = nuevos

        etiquetas = np.argmin(distancias[:, medoides], axis=1)
        self.medoid_indices_ = medoides
        self.cluster_centers_ = matriz[medoides]
        self.labels_ = etiquetas
        self.inertia_ = float(
            distancias[np.arange(len(matriz)), medoides[etiquetas]].sum()
        )
        return self

    def fit_predict(self, datos) -> np.ndarray:
        """Ajusta el modelo y retorna las etiquetas resultantes."""
        return self.fit(datos).labels_.copy()

    def predict(self, datos) -> np.ndarray:
        """Asigna nuevas observaciones al medoide más cercano."""
        if self.cluster_centers_ is None:
            raise ValueError("El modelo debe ajustarse antes de predecir.")
        matriz = np.asarray(datos, dtype=float)
        distancias = pairwise_distances(
            matriz, self.cluster_centers_, metric=self.metric
        )
        return np.argmin(distancias, axis=1)

    @staticmethod
    def _seleccionar_reemplazo(
        distancias: np.ndarray,
        medoides: np.ndarray,
        generador: np.random.Generator,
    ) -> int:
        """Elige un punto distante cuando un cluster queda vacío por empates."""
        disponibles = np.setdiff1d(np.arange(len(distancias)), medoides)
        if len(disponibles) == 0:
            return int(generador.integers(0, len(distancias)))
        distancia_actual = distancias[np.ix_(disponibles, medoides)].min(axis=1)
        return int(disponibles[np.argmax(distancia_actual)])
