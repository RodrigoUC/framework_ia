"""Visualizaciones de resultados independientes de Streamlit, en Plotly."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram as _dendrograma_scipy

from .results import (
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoParticion,
    ResultadoProyeccion,
)

COLOR_PRIMARIO = "#1A3C2B"
COLOR_SECUNDARIO = "#FF8C69"
COLOR_TERCIARIO = "#3C7D65"
COLOR_PUNTOS = "#2DD4BF"
PALETA_CUALITATIVA = px.colors.qualitative.T10


def _lang_vis() -> str:
    try:
        from ai_framework.ui.i18n import obtener_idioma
        return obtener_idioma()
    except Exception:
        return "es"



class VisualizadorRegresion:
    """Diagnósticos numéricos sobre evidencia guardada, sin recalcular modelos."""

    @staticmethod
    def real_predicho(resultado) -> go.Figure:
        limites = [
            min(resultado.y_true.min(), resultado.y_pred.min()),
            max(resultado.y_true.max(), resultado.y_pred.max()),
        ]
        figura = go.Figure(
            go.Scatter(
                x=resultado.y_true,
                y=resultado.y_pred,
                mode="markers",
                name="Prueba",
                marker={"color": COLOR_PUNTOS},
            )
        )
        figura.add_trace(
            go.Scatter(
                x=limites,
                y=limites,
                mode="lines",
                name="Predicción ideal",
                line={"color": COLOR_SECUNDARIO, "dash": "dash"},
            )
        )
        figura.update_layout(
            title="Valores reales y predichos · prueba",
            xaxis_title="Valor real",
            yaxis_title="Predicción",
        )
        return figura

    @staticmethod
    def residuos(resultado) -> go.Figure:
        figura = go.Figure(
            go.Scatter(
                x=resultado.y_pred,
                y=resultado.residuos,
                mode="markers",
                name="Residuos",
                marker={"color": COLOR_PUNTOS},
            )
        )
        figura.add_hline(y=0, line_dash="dash", line_color=COLOR_SECUNDARIO)
        figura.update_layout(
            title="Residuos · prueba",
            xaxis_title="Predicción",
            yaxis_title="Real − predicción",
        )
        return figura

    @staticmethod
    def comparacion_validacion(tabla: pd.DataFrame) -> go.Figure:
        disponibles = tabla.dropna(subset=["puntuacion_validacion"])
        return go.Figure(
            go.Bar(
                x=disponibles["algoritmo"] + " · " + disponibles["configuracion"],
                y=disponibles["puntuacion_validacion"],
                error_y={
                    "type": "data",
                    "array": disponibles["puntuacion_validacion_std"],
                },
                marker_color=COLOR_PRIMARIO,
            )
        ).update_layout(
            title="Comparación por validación",
            xaxis_title="Candidato",
            yaxis_title="RMSE medio (menor es mejor)",
        )


class VisualizadorNoSupervisado:
    """Fábrica de figuras Plotly para los resultados no supervisados."""

    @staticmethod
    def plano_acp(resultado: ResultadoACP) -> go.Figure:
        """Muestra las observaciones sobre los dos primeros componentes."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        datos = resultado.coordenadas
        x, y = datos.columns[:2]
        lang = _lang_vis()
        lbl_obs = "Observations" if lang == "en" else "Observaciones"
        titulo = "Principal plane of observations" if lang == "en" else "Plano principal de observaciones"
        eje_x = f"{x.replace('CP', 'PC')} ({resultado.varianza_explicada[x]:.2f} %)" if lang == "en" else f"{x} ({resultado.varianza_explicada[x]:.2f} %)"
        eje_y = f"{y.replace('CP', 'PC')} ({resultado.varianza_explicada[y]:.2f} %)" if lang == "en" else f"{y} ({resultado.varianza_explicada[y]:.2f} %)"

        figura = go.Figure(
            go.Scatter(
                x=datos[x],
                y=datos[y],
                mode="markers",
                name=lbl_obs,
                marker=dict(color=COLOR_PUNTOS, size=8, opacity=0.85),
            )
        )
        figura.add_hline(y=0, line_dash="dash", line_color="gray")
        figura.add_vline(x=0, line_dash="dash", line_color="gray")
        figura.update_layout(
            title=titulo,
            xaxis_title=eje_x,
            yaxis_title=eje_y,
        )
        return figura

    @staticmethod
    def circulo_correlacion(resultado: ResultadoACP, max_variables: int = 30) -> go.Figure:
        """Representa las cargas de las variables en los dos primeros ejes."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        cargas = resultado.cargas.iloc[:, :2].copy()
        if len(cargas) > max_variables:
            importancia = cargas.abs().sum(axis=1).nlargest(max_variables).index
            cargas = cargas.loc[importancia]

        figura = go.Figure()
        figura.add_shape(
            type="circle",
            x0=-1,
            y0=-1,
            x1=1,
            y1=1,
            line=dict(color=COLOR_PRIMARIO),
        )
        for variable, fila in cargas.iterrows():
            x, y = float(fila.iloc[0]), float(fila.iloc[1])
            figura.add_annotation(
                x=x * 0.95,
                y=y * 0.95,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLOR_PRIMARIO,
                arrowwidth=1.4,
            )
            figura.add_annotation(
                x=x * 1.08,
                y=y * 1.08,
                text=str(variable),
                showarrow=False,
                font=dict(size=10),
            )
        figura.add_hline(y=0, line_dash="dash", line_color="gray")
        figura.add_vline(x=0, line_dash="dash", line_color="gray")
        lang = _lang_vis()
        titulo = "Correlation circle" if lang == "en" else "Círculo de correlación"
        col_x = cargas.columns[0].replace("CP", "PC") if lang == "en" else cargas.columns[0]
        col_y = cargas.columns[1].replace("CP", "PC") if lang == "en" else cargas.columns[1]
        figura.update_xaxes(range=[-1.1, 1.1], title=col_x)
        figura.update_yaxes(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1, title=col_y)
        figura.update_layout(title=titulo)
        return figura

    @staticmethod
    def sobreposicion_acp(
        resultado: ResultadoACP, max_variables: int = 20
    ) -> go.Figure:
        """Superpone observaciones y vectores de variables sobre el círculo de correlación (Biplot)."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        coordenadas = resultado.coordenadas.iloc[:, :2].copy()
        cargas = resultado.cargas.iloc[:, :2].copy()
        if len(cargas) > max_variables:
            importancia = cargas.abs().sum(axis=1).nlargest(max_variables).index
            cargas = cargas.loc[importancia]

        escala = coordenadas.abs().max(axis=0).replace(0, 1)
        puntos = coordenadas.divide(escala, axis="columns")

        figura = go.Figure()

        figura.add_shape(
            type="circle",
            x0=-1,
            y0=-1,
            x1=1,
            y1=1,
            line=dict(color=COLOR_SECUNDARIO, width=1.5, dash="dash"),
        )

        lang = _lang_vis()
        lbl_obs = "Observations" if lang == "en" else "Observaciones"
        titulo = (
            "Superposition of observations and variables (PCA Biplot)"
            if lang == "en"
            else "Sobreposición de observaciones y variables (Biplot ACP)"
        )
        eje_x = coordenadas.columns[0].replace("CP", "PC") if lang == "en" else coordenadas.columns[0]
        eje_y = coordenadas.columns[1].replace("CP", "PC") if lang == "en" else coordenadas.columns[1]

        figura.add_trace(
            go.Scatter(
                x=puntos.iloc[:, 0],
                y=puntos.iloc[:, 1],
                mode="markers",
                marker=dict(color=COLOR_PUNTOS, size=7, opacity=0.85),
                name=lbl_obs,
                showlegend=True,
            )
        )

        for variable, fila in cargas.iterrows():
            x, y = float(fila.iloc[0]), float(fila.iloc[1])
            norma = np.hypot(x, y) or 1.0
            dx = (x / norma) * 0.08
            dy = (y / norma) * 0.08

            figura.add_annotation(
                x=x,
                y=y,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowcolor=COLOR_PRIMARIO,
                arrowwidth=1.6,
            )
            figura.add_annotation(
                x=x + dx,
                y=y + dy,
                text=str(variable),
                showarrow=False,
                xanchor="left" if x >= 0 else "right",
                yanchor="middle",
                font=dict(size=11),
            )

        figura.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        figura.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        figura.update_xaxes(
            range=[-1.25, 1.25],
            title=eje_x,
            zeroline=False,
        )
        figura.update_yaxes(
            range=[-1.25, 1.25],
            title=eje_y,
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
        )
        figura.update_layout(
            title=titulo,
            height=560,
            margin=dict(l=60, r=60, t=60, b=60),
        )
        return figura

    @staticmethod
    def varianza_acp(resultado: ResultadoACP) -> go.Figure:
        """Grafica varianza individual y acumulada por componente."""
        lang = _lang_vis()
        nom_ind = "Individual"
        nom_acu = "Cumulative" if lang == "en" else "Acumulada"
        titulo = "Variance explained by PCA" if lang == "en" else "Varianza explicada por el ACP"
        tit_y1 = "Explained variance (%)" if lang == "en" else "Varianza explicada (%)"
        tit_y2 = "Cumulative variance (%)" if lang == "en" else "Varianza acumulada (%)"
        tit_x = "Component" if lang == "en" else "Componente"

        tabla = pd.DataFrame(
            {
                nom_ind: resultado.varianza_explicada,
                nom_acu: resultado.varianza_acumulada,
            }
        )
        indices = [str(idx).replace("CP", "PC") for idx in tabla.index] if lang == "en" else tabla.index

        figura = make_subplots(specs=[[{"secondary_y": True}]])
        figura.add_trace(
            go.Bar(
                x=indices,
                y=tabla[nom_ind],
                name=nom_ind,
                marker_color=COLOR_TERCIARIO,
            ),
            secondary_y=False,
        )
        figura.add_trace(
            go.Scatter(
                x=indices,
                y=tabla[nom_acu],
                name=nom_acu,
                mode="lines+markers",
                line=dict(color=COLOR_SECUNDARIO, width=2),
            ),
            secondary_y=True,
        )
        figura.update_yaxes(title_text=tit_y1, secondary_y=False)
        figura.update_yaxes(title_text=tit_y2, range=[0, 105], secondary_y=True)
        figura.update_xaxes(title_text=tit_x)
        figura.update_layout(title=titulo)
        return figura

    @staticmethod
    def clusters(resultado: ResultadoCluster) -> go.Figure:
        """Colorea la proyección bidimensional con las etiquetas encontradas."""
        datos = resultado.proyeccion_2d.copy()
        datos["cluster"] = resultado.etiquetas.astype(str)
        columnas = resultado.proyeccion_2d.columns
        figura = px.scatter(
            datos,
            x=columnas[0],
            y=columnas[1],
            color="cluster",
            color_discrete_sequence=PALETA_CUALITATIVA,
            opacity=0.78,
        )
        figura.update_traces(marker=dict(size=9))
        figura.update_layout(
            title=f"{resultado.algoritmo}: proyección de los clusters",
            legend_title="Cluster",
        )
        return figura

    @staticmethod
    def perfiles_cluster(resultado: ResultadoCluster, max_variables: int = 25) -> go.Figure:
        """Compara los centroides o perfiles promedio mediante un mapa de calor."""
        perfiles = resultado.centroides.copy()
        if perfiles.shape[1] > max_variables:
            variables = perfiles.var(axis=0).nlargest(max_variables).index
            perfiles = perfiles.loc[:, variables]
        figura = px.imshow(
            perfiles,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            aspect="auto",
        )
        figura.update_layout(
            title=f"Perfiles de {resultado.algoritmo}",
            xaxis_title="Variable transformada",
            yaxis_title="Cluster",
            height=max(300, len(perfiles) * 60),
        )
        return figura

    @staticmethod
    def dendrograma(resultado: ResultadoCluster, max_etiquetas: int = 80) -> go.Figure:
        """Genera el dendrograma de un resultado jerárquico."""
        if resultado.matriz_vinculacion is None:
            raise ValueError("El resultado no contiene una matriz de vinculación.")
        cantidad = len(resultado.etiquetas)
        mostrar_etiquetas = cantidad <= max_etiquetas
        etiquetas = (
            resultado.etiquetas.index.astype(str).tolist()
            if mostrar_etiquetas
            else None
        )
        info = _dendrograma_scipy(
            resultado.matriz_vinculacion, labels=etiquetas, no_plot=True
        )
        figura = go.Figure()
        for x, y in zip(info["icoord"], info["dcoord"]):
            figura.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=COLOR_PRIMARIO, width=1.4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if mostrar_etiquetas:
            posiciones = list(range(5, 5 + 10 * len(info["ivl"]), 10))
            figura.update_xaxes(
                tickmode="array",
                tickvals=posiciones,
                ticktext=info["ivl"],
                tickangle=90,
            )
        else:
            figura.update_xaxes(showticklabels=False)
        figura.update_layout(
            title=f"Dendrograma {resultado.algoritmo}",
            xaxis_title="Observaciones",
            yaxis_title="Distancia",
        )
        return figura

    @staticmethod
    def curva_evaluacion(tabla: pd.DataFrame, algoritmo: str) -> go.Figure:
        """Grafica silhouette y, cuando existe, la inercia del benchmark."""
        if "metodo" in tabla.columns:
            figura = px.line(
                tabla,
                x="k",
                y="silhouette",
                color="metodo",
                markers=True,
                color_discrete_sequence=PALETA_CUALITATIVA,
            )
        else:
            figura = px.line(
                tabla,
                x="k",
                y="silhouette",
                markers=True,
                color_discrete_sequence=[COLOR_PRIMARIO],
            )
        figura.update_layout(
            title=f"Evaluación de {algoritmo}",
            xaxis_title="Número de clusters",
            yaxis_title="Silhouette",
        )
        return figura

    @staticmethod
    def proyeccion(resultado: ResultadoProyeccion, etiquetas=None) -> go.Figure:
        """Grafica una proyección t-SNE o UMAP, opcionalmente coloreada."""
        datos = resultado.coordenadas.copy()
        columnas = datos.columns
        if etiquetas is None:
            figura = px.scatter(datos, x=columnas[0], y=columnas[1], opacity=0.75)
            figura.update_traces(marker=dict(color=COLOR_PUNTOS, size=8))
        else:
            datos["grupo"] = pd.Series(etiquetas, index=datos.index).astype(str)
            figura = px.scatter(
                datos,
                x=columnas[0],
                y=columnas[1],
                color="grupo",
                color_discrete_sequence=PALETA_CUALITATIVA,
                opacity=0.78,
            )
            figura.update_traces(marker=dict(size=9))
        figura.update_layout(title=f"Proyección {resultado.algoritmo}")
        return figura

    @staticmethod
    def _validar_acp_2d(resultado: ResultadoACP) -> None:
        """Garantiza que la visualización tenga dos componentes disponibles."""
        if resultado.coordenadas.shape[1] < 2:
            raise ValueError("La visualización requiere al menos dos componentes.")


class VisualizadorSupervisado(VisualizadorRegresion):
    """Visualizaciones Plotly unificadas para flujos supervisados (clasificación y regresión)."""

    @staticmethod
    def matriz_confusion(resultado: ResultadoClasificacion) -> go.Figure:
        """Dibuja una matriz de confusión con formato de tabla."""
        figura = px.imshow(
            resultado.matriz_confusion,
            text_auto="d",
            color_continuous_scale="Blues",
            aspect="auto",
        )
        figura.update_layout(
            title=f"Matriz de confusión — {resultado.algoritmo}",
            xaxis_title="Predicción",
            yaxis_title="Real",
        )
        return figura

    @staticmethod
    def metricas_barras(resultado: ResultadoClasificacion) -> go.Figure | None:
        """Compara precisión, recall y F1 por clase."""
        metricas = resultado.metricas.get("precision", {}).get("por_clase", {})
        if not metricas:
            return None

        etiquetas = list(metricas.keys())
        frame = pd.DataFrame(
            {
                "Precision": [valor["precision"] for valor in metricas.values()],
                "Recall": [valor["recall"] for valor in metricas.values()],
                "F1": [valor["f1"] for valor in metricas.values()],
            },
            index=etiquetas,
        )
        figura = go.Figure()
        for columna, color in zip(frame.columns, PALETA_CUALITATIVA):
            figura.add_trace(go.Bar(x=frame.index, y=frame[columna], name=columna, marker_color=color))
        figura.update_layout(
            title="Métricas por clase",
            xaxis_title="Clase",
            yaxis_title="Valor",
            yaxis_range=[0, 1.05],
            barmode="group",
        )
        return figura

    @staticmethod
    def distribucion_estratificada(resultado: ResultadoClasificacion) -> go.Figure | None:
        """Compara la distribución de clases en el conjunto real y predicho."""
        metricas = resultado.metricas.get("distribucion", {})
        reales = pd.Series(metricas.get("real", {}))
        predichos = pd.Series(metricas.get("predicho", {}))
        entrenamiento = pd.Series(metricas.get("train", {}))
        if reales.empty and predichos.empty:
            return None
        marco = pd.DataFrame(
            {"Train": entrenamiento, "Real": reales, "Predicho": predichos}
        ).fillna(0)
        figura = go.Figure()
        for columna, color in zip(marco.columns, PALETA_CUALITATIVA):
            figura.add_trace(go.Bar(x=marco.index, y=marco[columna], name=columna, marker_color=color))
        figura.update_layout(
            title="Distribución estratificada en test",
            xaxis_title="Clase",
            yaxis_title="Frecuencia",
            barmode="group",
        )
        return figura


class VisualizadorDatos:
    """Visualizaciones Plotly para la vista de Datos y preparación."""

    @staticmethod
    def tipos_columnas(tabla: pd.DataFrame) -> go.Figure:
        """Grafica la cantidad de columnas numéricas frente a categóricas."""
        conteo = tabla["tipo"].value_counts()
        try:
            from ai_framework.ui.i18n import obtener_idioma
            lang = obtener_idioma()
        except Exception:
            lang = "es"
        if lang == "en":
            etiquetas_map = {"Numérica": "Numerical", "Categórica": "Categorical"}
            nombres = [etiquetas_map.get(k, k) for k in conteo.index]
            figura = px.pie(
                names=nombres,
                values=conteo.values,
                color=nombres,
                color_discrete_map={
                    "Numerical": COLOR_PRIMARIO,
                    "Categorical": COLOR_SECUNDARIO,
                    "Numérica": COLOR_PRIMARIO,
                    "Categórica": COLOR_SECUNDARIO,
                },
                hole=0.45,
            )
            figura.update_traces(textinfo="label+value")
            figura.update_layout(title="Column types")
            return figura
        figura = px.pie(
            names=conteo.index,
            values=conteo.values,
            color=conteo.index,
            color_discrete_map={"Numérica": COLOR_PRIMARIO, "Categórica": COLOR_SECUNDARIO},
            hole=0.45,
        )
        figura.update_traces(textinfo="label+value")
        figura.update_layout(title="Tipos de columna")
        return figura

    @staticmethod
    def tamanos_particion(resultado: ResultadoParticion) -> go.Figure:
        """Compara el tamaño de train, test y validación (si existe)."""
        conteos = {"Train": len(resultado.train), "Test": len(resultado.test)}
        if resultado.validacion is not None:
            conteos["Validación"] = len(resultado.validacion)
        figura = go.Figure(
            go.Bar(
                x=list(conteos.keys()),
                y=list(conteos.values()),
                marker_color=[COLOR_PRIMARIO, COLOR_SECUNDARIO, COLOR_TERCIARIO][: len(conteos)],
                text=list(conteos.values()),
                textposition="outside",
            )
        )
        figura.update_layout(
            title="Tamaño de la partición", xaxis_title="Subconjunto", yaxis_title="Filas"
        )
        return figura

    @staticmethod
    def distribucion_particion(resultado: ResultadoParticion) -> go.Figure | None:
        """Compara la distribución de la columna de estratificación por subconjunto."""
        if resultado.columna_estratificacion is None:
            return None
        subconjuntos = {"Train": resultado.distribucion_train, "Test": resultado.distribucion_test}
        if resultado.distribucion_validacion is not None:
            subconjuntos["Validación"] = resultado.distribucion_validacion
        marco = pd.DataFrame(subconjuntos).fillna(0)
        figura = go.Figure()
        for columna, color in zip(marco.columns, PALETA_CUALITATIVA):
            figura.add_trace(go.Bar(x=marco.index.astype(str), y=marco[columna], name=columna, marker_color=color))
        figura.update_layout(
            title=f"Distribución de '{resultado.columna_estratificacion}' por subconjunto",
            xaxis_title=resultado.columna_estratificacion,
            yaxis_title="Frecuencia",
            barmode="group",
        )
        return figura


SupervisedVisualizer = VisualizadorSupervisado
ClassificationVisualizer = VisualizadorSupervisado
DataVisualizer = VisualizadorDatos
UnsupervisedVisualizer = VisualizadorNoSupervisado
RegressionVisualizer = VisualizadorRegresion
