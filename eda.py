"""
EDA — Análisis Exploratorio de Datos (clase base del framework).

Este módulo contiene ÚNICAMENTE la lógica de análisis y preparación de datos.
La parte gráfica (Visualización con Streamlit) vive en un archivo separado
(``vca.py``), tal como pide el proyecto: **streamlit es la parte gráfica**.

Responsabilidades:
    DataFrame.py  -> operaciones tabulares y de EDA desarrolladas
    eda.py        -> clase EDA: pipeline que deja el dataset ordenado
    vca.py        -> interfaz gráfica (Streamlit) del EDA

Hereda de :class:`DataFrame` (misma carpeta).

Jerarquía del framework:

    EDA               (este módulo · solo lógica)
    ├── NoSupervisado (no_supervisado.py)
    │      ├── ReduccionDimensional -> ACP, t-SNE, UMAP
    │      └── Cluster -> K-Means, K-Medoids, HAC
    └── Supervisado   (supervisado.py)
           ├── Clasificacion  ->  RF, NR
           └── Progresion     ->  RLS, RLM, RL
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from DataFrame import DataFrame as DataFrameBase


class EDA(DataFrameBase):
    """Base del análisis exploratorio de datos.

    Hereda :class:`DataFrame` y agrega un pipeline de preparación que deja
    el dataset ordenado para el análisis posterior (cluster, clasificación
    o progresión). No contiene lógica de Streamlit; la interfaz gráfica se
    encuentra en :mod:`vca`.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        ruta_datos: str | Path | None = None,
        *,
        separador: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(dataframe=dataframe)
        if ruta_datos is not None:
            self.cargar_csv(ruta_datos, separador=separador, encoding=encoding)

    # ==========================================================
    # Preparación del dataset
    # ==========================================================

    def preparar_dataset(
        self,
        *,
        eliminar_duplicados: bool = True,
        imputar_nulos: bool = True,
        normalizar: bool = False,
        estandarizar: bool = False,
    ) -> pd.DataFrame:
        """Prepara el dataset y lo deja ordenado para el análisis posterior."""
        if self.datos.empty:
            raise ValueError("El DataFrame está vacío.")

        if eliminar_duplicados:
            self.eliminar_duplicados()
        if imputar_nulos:
            self.imputar_nulos()
        if normalizar and estandarizar:
            raise ValueError("No se puede normalizar y estandarizar a la vez.")
        if normalizar:
            self.normalizar()
        if estandarizar:
            self.estandarizar()

        self.datos = self.datos.reset_index(drop=True)
        return self.datos

    def resumen_calidad(self) -> dict[str, int | float]:
        """Indicadores sintéticos de calidad del dataset actual."""
        datos = self.datos
        return {
            "filas": int(len(datos)),
            "columnas": int(datos.shape[1]),
            "duplicados": int(datos.duplicated().sum()),
            "nulos": int(datos.isna().sum().sum()),
            "porcentaje_nulos": round(float(datos.isna().mean().mean() * 100), 2),
            "numericas": int(datos.select_dtypes(include="number").shape[1]),
            "categoricas": int(datos.select_dtypes(exclude="number").shape[1]),
        }
