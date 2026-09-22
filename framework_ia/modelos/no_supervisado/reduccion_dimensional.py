"""Algoritmos de reducción dimensional separados del agrupamiento."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .base import NoSupervisado
from ...resultados import ResultadoACP, ResultadoProyeccion


class DependenciaOpcionalError(ImportError):
    """Indica que una capacidad opcional requiere instalar una dependencia."""


class ReduccionDimensional(NoSupervisado):
    """Ejecuta ACP, t-SNE y UMAP sobre un preprocesamiento común."""

    @staticmethod
    def _configurar_cache_numba() -> Path:
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

    def ajustar(self, algoritmo: str = "ACP", **kwargs):
        """Despacha el algoritmo solicitado manteniendo una interfaz uniforme."""
        metodos = {
            "ACP": self.ACP,
            "TSNE": self.TSNE,
            "UMAP": self.UMAP,
        }
        try:
            metodo = metodos[algoritmo.upper()]
        except KeyError as exc:
            raise ValueError(f"Algoritmo no soportado: {algoritmo}") from exc
        return metodo(**kwargs)

    def ACP(self, n_componentes: int = 2) -> ResultadoACP:
        """Calcula componentes, cargas y varianza explicada con sklearn."""
        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        maximo = min(matriz.shape)
        if not 1 <= n_componentes <= maximo:
            raise ValueError(
                f"n_componentes debe estar entre 1 y {maximo}."
            )

        modelo = PCA(n_components=n_componentes)
        valores = modelo.fit_transform(matriz)
        nombres = [f"CP{i + 1}" for i in range(n_componentes)]
        coordenadas = pd.DataFrame(valores, index=matriz.index, columns=nombres)

        cargas_np = modelo.components_.T * np.sqrt(modelo.explained_variance_)
        cargas = pd.DataFrame(cargas_np, index=matriz.columns, columns=nombres)
        varianza = pd.Series(
            modelo.explained_variance_ratio_ * 100,
            index=nombres,
            name="varianza_explicada_porcentaje",
        )
        acumulada = varianza.cumsum().rename("varianza_acumulada_porcentaje")
        self.modelo = modelo
        return ResultadoACP(
            coordenadas=coordenadas,
            cargas=cargas,
            varianza_explicada=varianza,
            varianza_acumulada=acumulada,
            datos_preparados=preparados,
            modelo=modelo,
        )

    def acp(self, n_componentes: int = 2) -> ResultadoACP:
        """Alias pythonico de :meth:`ACP`."""
        return self.ACP(n_componentes=n_componentes)

    def TSNE(
        self,
        *,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Genera una proyección t-SNE bidimensional."""
        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        if len(matriz) < 3:
            raise ValueError("t-SNE requiere al menos tres filas.")
        if not 1 <= perplexity < len(matriz):
            raise ValueError(
                f"perplexity debe ser menor que el número de filas ({len(matriz)})."
            )

        parametros = {
            "n_components": 2,
            "perplexity": perplexity,
            "init": "pca",
            "learning_rate": "auto",
            "random_state": random_state,
        }
        try:
            # scikit-learn >= 1.5 renombró n_iter a max_iter.
            modelo = TSNE(max_iter=max_iter, **parametros)
        except TypeError as exc:
            if "max_iter" not in str(exc):
                raise
            modelo = TSNE(n_iter=max_iter, **parametros)
        valores = modelo.fit_transform(matriz)
        coordenadas = pd.DataFrame(
            valores,
            index=matriz.index,
            columns=["TSNE1", "TSNE2"],
        )
        self.modelo = modelo
        return ResultadoProyeccion(
            algoritmo="t-SNE",
            coordenadas=coordenadas,
            datos_preparados=preparados,
            modelo=modelo,
        )

    def tsne(
        self,
        *,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Alias pythonico de :meth:`TSNE`."""
        return self.TSNE(
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )

    def UMAP(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Genera una proyección UMAP bidimensional y reproducible.

        ``umap-learn`` usa Numba durante la importación. En instalaciones donde
        el directorio de caché por defecto no es escribible (por ejemplo, una
        sesión de laboratorio o un contenedor), Numba aborta antes de crear el
        modelo. Si la ruta configurada no es escribible, se usa un caché
        temporal local; de otro modo se conserva la configuración explícita.
        """
        self._configurar_cache_numba()
        try:
            import umap.umap_ as umap
        except (ImportError, RuntimeError) as exc:
            raise DependenciaOpcionalError(
                "No fue posible inicializar UMAP. Instale 'umap-learn' y "
                "verifique que Numba tenga un directorio de caché escribible. "
                f"Detalle técnico: {exc}"
            ) from exc

        preparados = self.preparar_matriz()
        matriz = preparados.matriz
        if len(matriz) < 3:
            raise ValueError("UMAP requiere al menos tres filas.")
        if not 2 <= n_neighbors < len(matriz):
            raise ValueError(
                f"n_neighbors debe estar entre 2 y {len(matriz) - 1}."
            )

        modelo = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=1,
        )
        valores = modelo.fit_transform(matriz)
        coordenadas = pd.DataFrame(
            valores,
            index=matriz.index,
            columns=["UMAP1", "UMAP2"],
        )
        self.modelo = modelo
        return ResultadoProyeccion(
            algoritmo="UMAP",
            coordenadas=coordenadas,
            datos_preparados=preparados,
            modelo=modelo,
        )

    def umap(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> ResultadoProyeccion:
        """Alias pythonico de :meth:`UMAP`."""
        return self.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
