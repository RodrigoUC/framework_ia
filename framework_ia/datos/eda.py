"""
EDA — Análisis Exploratorio de Datos (clase base del framework).

Este módulo contiene ÚNICAMENTE la lógica de análisis y preparación de datos.
La interfaz Streamlit se implementa en ``ui/streamlit_app.py``.

Responsabilidades:
    datos/dataframe.py  -> operaciones tabulares y estadísticas
    datos/eda.py        -> clase EDA: pipeline de preparación y gráficos del EDA
    utils/              -> funciones y objetos de configuración
    ui/streamlit_app.py -> interfaz gráfica (Streamlit) del EDA

Hereda de :class:`DataFrame` (misma carpeta).

Jerarquía del framework:

    EDA               (este módulo · lógica y gráficos del EDA)
    ├── NoSupervisado (modelos/no_supervisado/)
    │      └── Cluster -> ACP, K-Means (+ t-SNE, UMAP), K-Medoids, HAC
    ├── Supervisado   (modelos/supervisado.py)
    │      ├── Clasificacion  ->  RF, NR
    │      └── Regresion      ->  RLS, RLM, RL
    └── Pizarra       (agentes/ · estado que comparten los agentes)

    Agente (agentes/base.py, ABC) trabaja sobre una Pizarra:
           ├── AgenteEDA         ->  perfil, nulos, atípicos, fuga, desbalance
           └── AgenteClustering  ->  compara K-Means y HAC, recomienda k
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from .dataframe import DataFrame as DataFrameBase


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
        self.preparacion: dict | None = None
        if ruta_datos is not None:
            self.cargar_csv(ruta_datos, separador=separador, encoding=encoding)

    # ==========================================================
    # Preparación del dataset
    # ==========================================================

    def preparar_dataset(
        self,
        *,
        columnas_eliminar: list[str] | tuple[str, ...] | None = None,
        columnas_sin_nulos: list[str] | tuple[str, ...] | None = None,
        eliminar_duplicados: bool = True,
        imputar_nulos: bool = True,
        normalizar: bool = False,
        estandarizar: bool = False,
    ) -> pd.DataFrame:
        """Filtra filas sólo en columnas explícitas; la imputación conserva su API."""
        if self.datos.empty:
            raise ValueError("El DataFrame está vacío.")

        filas_originales = len(self.datos)
        if columnas_eliminar:
            self.eliminar_columnas(columnas_eliminar)
        columnas_sin_nulos = list(dict.fromkeys(columnas_sin_nulos or ()))
        inexistentes = [c for c in columnas_sin_nulos if c not in self.datos]
        if inexistentes:
            raise KeyError(f"Columnas para filtrar nulos inexistentes: {inexistentes}")
        filtrado = self.datos.dropna(subset=columnas_sin_nulos)
        if filtrado.empty:
            raise ValueError(
                "El filtro de nulos eliminaría todas las filas. Revise las columnas seleccionadas."
            )
        filas_eliminadas_nulos = len(self.datos) - len(filtrado)
        self.datos = filtrado.copy()
        antes_duplicados = len(self.datos)
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
        self.preparacion = {
            "columnas_eliminar": list(columnas_eliminar or ()),
            "columnas_sin_nulos": columnas_sin_nulos,
            "eliminar_duplicados": eliminar_duplicados,
            "filas_originales": filas_originales,
            "filas_eliminadas_nulos": filas_eliminadas_nulos,
            "filas_eliminadas_duplicados": antes_duplicados - len(self.datos),
            "filas_preparadas": len(self.datos),
        }
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

    def balance_objetivo(self, objetivo: str) -> tuple[pd.DataFrame, go.Figure | None]:
        """Return class counts/proportions and an EDA chart for an active target."""
        if objetivo not in self.datos.columns:
            raise KeyError(f"No existe la columna objetivo {objetivo!r}.")
        valores = self.datos[objetivo]
        conteos = valores.value_counts(dropna=True, sort=False).rename("Cantidad")
        conteos = conteos.loc[conteos > 0]
        resumen = conteos.to_frame()
        resumen["Proporción"] = resumen["Cantidad"] / max(int(valores.notna().sum()), 1)
        if resumen.empty:
            return resumen, None
        figura = go.Figure(
            go.Bar(
                x=resumen.index.tolist(),
                y=resumen["Cantidad"].tolist(),
                marker_color="#1A3C2B",
            )
        )
        figura.update_layout(
            title=f"Distribución de clases de {objetivo}",
            xaxis={"title": objetivo, "type": "category"},
            yaxis_title="Cantidad",
        )
        return resumen, figura

    # ==========================================================
    # Gráficos del EDA
    # ==========================================================

    def histogramas(self, columnas=None, mostrar: bool = True) -> go.Figure:
        """Histogramas (con KDE) de cada variable numérica, en Plotly."""
        numericas = self._seleccionar_numericas(columnas)
        cantidad = len(numericas)
        if cantidad == 0:
            raise ValueError("No hay columnas numéricas para graficar.")

        columnas_grilla = min(cantidad, 3)
        filas = int(np.ceil(cantidad / columnas_grilla))
        figura = make_subplots(
            rows=filas,
            cols=columnas_grilla,
            subplot_titles=[f"Distribución de {columna}" for columna in numericas],
        )
        for indice, columna in enumerate(numericas):
            fila, col = divmod(indice, columnas_grilla)
            valores = self.datos[columna].dropna()
            figura.add_trace(
                go.Histogram(
                    x=valores,
                    histnorm="probability density",
                    marker_color="#1A3C2B",
                    showlegend=False,
                ),
                row=fila + 1,
                col=col + 1,
            )
            if valores.nunique() > 1:
                densidad = gaussian_kde(valores)
                rango = np.linspace(valores.min(), valores.max(), 200)
                figura.add_trace(
                    go.Scatter(
                        x=rango,
                        y=densidad(rango),
                        mode="lines",
                        line=dict(color="#FF8C69", width=2),
                        showlegend=False,
                    ),
                    row=fila + 1,
                    col=col + 1,
                )
        figura.update_layout(height=320 * filas, margin=dict(t=60))
        if mostrar:
            figura.show()
        return figura

    def boxplots(self, columnas=None, mostrar: bool = True) -> go.Figure:
        """Diagramas de caja de cada variable numérica, en Plotly."""
        numericas = self._seleccionar_numericas(columnas)
        cantidad = len(numericas)
        if cantidad == 0:
            raise ValueError("No hay columnas numéricas para graficar.")

        columnas_grilla = min(cantidad, 3)
        filas = int(np.ceil(cantidad / columnas_grilla))
        figura = make_subplots(
            rows=filas,
            cols=columnas_grilla,
            subplot_titles=[f"Diagrama de caja de {columna}" for columna in numericas],
        )
        for indice, columna in enumerate(numericas):
            fila, col = divmod(indice, columnas_grilla)
            figura.add_trace(
                go.Box(
                    y=self.datos[columna],
                    name=columna,
                    marker_color="#7DBE76",
                    showlegend=False,
                ),
                row=fila + 1,
                col=col + 1,
            )
        figura.update_layout(height=300 * filas, margin=dict(t=60))
        if mostrar:
            figura.show()
        return figura

    def scatterplots(self, columnas=None, mostrar: bool = True) -> go.Figure:
        """Matriz de dispersión entre variables numéricas, en Plotly."""
        numericas = self._seleccionar_numericas(columnas)
        if len(numericas) < 2:
            raise ValueError("Se requieren al menos dos columnas numéricas.")
        figura = px.scatter_matrix(self.datos[numericas], dimensions=numericas)
        figura.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(color="#1A3C2B", opacity=0.7))
        figura.update_layout(height=max(500, 220 * len(numericas)))
        if mostrar:
            figura.show()
        return figura

    def mapa_calor(
        self, columnas=None, metodo: str = "pearson", mostrar: bool = True
    ) -> tuple[go.Figure, pd.DataFrame]:
        """Grafica y retorna la matriz de correlación de variables numéricas."""
        numericas = self._seleccionar_numericas(columnas)
        correlacion = self.datos[numericas].corr(method=metodo).round(3)
        figura = px.imshow(
            correlacion,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title=f"Correlación ({metodo})",
        )
        figura.update_layout(height=max(400, 60 * len(numericas)))
        if mostrar:
            figura.show()
        return figura, correlacion
