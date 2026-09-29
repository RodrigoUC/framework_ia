"""Visualizaciones de resultados independientes de Streamlit, en Plotly."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram as _dendrograma_scipy

from .resultados import (
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoParticion,
    ResultadoProyeccion,
)

COLOR_PRIMARIO = "#1A3C2B"
COLOR_SECUNDARIO = "#FF8C69"
COLOR_TERCIARIO = "#3C7D65"
PALETA_CUALITATIVA = px.colors.qualitative.T10


class VisualizadorNoSupervisado:
    """Fábrica de figuras Plotly para los resultados no supervisados."""

    @staticmethod
    def plano_acp(resultado: ResultadoACP) -> go.Figure:
        """Muestra las observaciones sobre los dos primeros componentes."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        datos = resultado.coordenadas
        x, y = datos.columns[:2]
        figura = go.Figure(
            go.Scatter(
                x=datos[x],
                y=datos[y],
                mode="markers",
                marker=dict(color=COLOR_PRIMARIO, size=8, opacity=0.75),
            )
        )
        figura.add_hline(y=0, line_dash="dash", line_color="gray")
        figura.add_vline(x=0, line_dash="dash", line_color="gray")
        figura.update_layout(
            title="Plano principal de observaciones",
            xaxis_title=f"{x} ({resultado.varianza_explicada[x]:.2f} %)",
            yaxis_title=f"{y} ({resultado.varianza_explicada[y]:.2f} %)",
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
        figura.update_xaxes(range=[-1.1, 1.1])
        figura.update_yaxes(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1)
        figura.update_layout(title="Círculo de correlación")
        return figura

    @staticmethod
    def sobreposicion_acp(resultado: ResultadoACP, max_variables: int = 20) -> go.Figure:
        """Superpone observaciones y cargas en un biplot comparable."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        coordenadas = resultado.coordenadas.iloc[:, :2].copy()
        cargas = resultado.cargas.iloc[:, :2].copy()
        if len(cargas) > max_variables:
            importancia = cargas.abs().sum(axis=1).nlargest(max_variables).index
            cargas = cargas.loc[importancia]

        escala = coordenadas.abs().max(axis=0).replace(0, 1)
        puntos = coordenadas.divide(escala, axis="columns")
        figura = go.Figure(
            go.Scatter(
                x=puntos.iloc[:, 0],
                y=puntos.iloc[:, 1],
                mode="markers",
                marker=dict(color="gray", size=7, opacity=0.55),
                showlegend=False,
            )
        )
        for variable, fila in cargas.iterrows():
            x, y = float(fila.iloc[0]), float(fila.iloc[1])
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
        figura.update_xaxes(range=[-1.15, 1.15], title=coordenadas.columns[0])
        figura.update_yaxes(range=[-1.15, 1.15], title=coordenadas.columns[1])
        figura.update_layout(title="Sobreposición de observaciones y variables")
        return figura

    @staticmethod
    def varianza_acp(resultado: ResultadoACP) -> go.Figure:
        """Grafica varianza individual y acumulada por componente."""
        tabla = pd.DataFrame(
            {
                "Individual": resultado.varianza_explicada,
                "Acumulada": resultado.varianza_acumulada,
            }
        )
        figura = make_subplots(specs=[[{"secondary_y": True}]])
        figura.add_trace(
            go.Bar(
                x=tabla.index,
                y=tabla["Individual"],
                name="Individual",
                marker_color=COLOR_TERCIARIO,
            ),
            secondary_y=False,
        )
        figura.add_trace(
            go.Scatter(
                x=tabla.index,
                y=tabla["Acumulada"],
                name="Acumulada",
                mode="lines+markers",
                line=dict(color=COLOR_SECUNDARIO, width=2),
            ),
            secondary_y=True,
        )
        figura.update_yaxes(title_text="Varianza explicada (%)", secondary_y=False)
        figura.update_yaxes(title_text="Varianza acumulada (%)", range=[0, 105], secondary_y=True)
        figura.update_xaxes(title_text="Componente")
        figura.update_layout(title="Varianza explicada por el ACP")
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
            figura.update_traces(marker=dict(color=COLOR_PRIMARIO, size=8))
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


class VisualizadorSupervisado:
    """Visualizaciones Plotly para flujos supervisados."""

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
