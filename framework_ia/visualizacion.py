"""Visualizaciones de resultados independientes de Streamlit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from .resultados import (
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoProyeccion,
)


class VisualizadorNoSupervisado:
    """Fábrica de figuras Matplotlib para los resultados del framework."""

    @staticmethod
    def plano_acp(resultado: ResultadoACP):
        """Muestra las observaciones sobre los dos primeros componentes."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        figura, eje = plt.subplots(figsize=(9, 6))
        datos = resultado.coordenadas
        eje.scatter(datos.iloc[:, 0], datos.iloc[:, 1], alpha=0.72, s=35)
        eje.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        eje.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        x, y = datos.columns[:2]
        eje.set_xlabel(f"{x} ({resultado.varianza_explicada[x]:.2f} %)")
        eje.set_ylabel(f"{y} ({resultado.varianza_explicada[y]:.2f} %)")
        eje.set_title("Plano principal de observaciones")
        figura.tight_layout()
        return figura

    @staticmethod
    def circulo_correlacion(resultado: ResultadoACP, max_variables: int = 30):
        """Representa las cargas de las variables en los dos primeros ejes."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        cargas = resultado.cargas.iloc[:, :2].copy()
        if len(cargas) > max_variables:
            importancia = cargas.abs().sum(axis=1).nlargest(max_variables).index
            cargas = cargas.loc[importancia]

        figura, eje = plt.subplots(figsize=(9, 8))
        circulo = plt.Circle((0, 0), 1, fill=False, color="#1A3C2B")
        eje.add_patch(circulo)
        for variable, fila in cargas.iterrows():
            x, y = float(fila.iloc[0]), float(fila.iloc[1])
            eje.arrow(
                0,
                0,
                x * 0.95,
                y * 0.95,
                color="#1A3C2B",
                alpha=0.65,
                head_width=0.025,
                length_includes_head=True,
            )
            eje.text(x * 1.06, y * 1.06, variable, fontsize=8, ha="center")
        eje.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        eje.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        eje.set_xlim(-1.1, 1.1)
        eje.set_ylim(-1.1, 1.1)
        eje.set_aspect("equal")
        eje.set_title("Círculo de correlación")
        figura.tight_layout()
        return figura

    @staticmethod
    def sobreposicion_acp(resultado: ResultadoACP, max_variables: int = 20):
        """Superpone observaciones y cargas en un biplot comparable."""
        VisualizadorNoSupervisado._validar_acp_2d(resultado)
        coordenadas = resultado.coordenadas.iloc[:, :2].copy()
        cargas = resultado.cargas.iloc[:, :2].copy()
        if len(cargas) > max_variables:
            importancia = cargas.abs().sum(axis=1).nlargest(max_variables).index
            cargas = cargas.loc[importancia]

        escala = coordenadas.abs().max(axis=0).replace(0, 1)
        puntos = coordenadas.divide(escala, axis="columns")
        figura, eje = plt.subplots(figsize=(9, 8))
        eje.scatter(
            puntos.iloc[:, 0],
            puntos.iloc[:, 1],
            color="gray",
            alpha=0.55,
            s=28,
        )
        for variable, fila in cargas.iterrows():
            x, y = float(fila.iloc[0]), float(fila.iloc[1])
            eje.arrow(
                0,
                0,
                x,
                y,
                color="#1A3C2B",
                alpha=0.7,
                head_width=0.025,
                length_includes_head=True,
            )
            eje.text(x * 1.08, y * 1.08, variable, fontsize=8, ha="center")
        eje.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        eje.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        eje.set_xlim(-1.15, 1.15)
        eje.set_ylim(-1.15, 1.15)
        eje.set_title("Sobreposición de observaciones y variables")
        eje.set_xlabel(coordenadas.columns[0])
        eje.set_ylabel(coordenadas.columns[1])
        figura.tight_layout()
        return figura

    @staticmethod
    def varianza_acp(resultado: ResultadoACP):
        """Grafica varianza individual y acumulada por componente."""
        tabla = pd.DataFrame(
            {
                "Individual": resultado.varianza_explicada,
                "Acumulada": resultado.varianza_acumulada,
            }
        )
        figura, eje = plt.subplots(figsize=(9, 5))
        tabla["Individual"].plot.bar(ax=eje, color="#3C7D65", alpha=0.8)
        eje.set_ylabel("Varianza explicada (%)")
        eje.set_xlabel("Componente")
        eje.set_title("Varianza explicada por el ACP")
        eje_secundario = eje.twinx()
        tabla["Acumulada"].plot(
            ax=eje_secundario, color="#FF8C69", marker="o", linewidth=2
        )
        eje_secundario.set_ylim(0, 105)
        eje_secundario.set_ylabel("Varianza acumulada (%)")
        figura.tight_layout()
        return figura

    @staticmethod
    def clusters(resultado: ResultadoCluster):
        """Colorea la proyección bidimensional con las etiquetas encontradas."""
        datos = resultado.proyeccion_2d.copy()
        datos["cluster"] = resultado.etiquetas.astype(str)
        figura, eje = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=datos,
            x=datos.columns[0],
            y=datos.columns[1],
            hue="cluster",
            palette="tab10",
            s=48,
            alpha=0.78,
            ax=eje,
        )
        eje.set_title(f"{resultado.algoritmo}: proyección de los clusters")
        eje.legend(title="Cluster")
        figura.tight_layout()
        return figura

    @staticmethod
    def perfiles_cluster(resultado: ResultadoCluster, max_variables: int = 25):
        """Compara los centroides o perfiles promedio mediante un mapa de calor."""
        perfiles = resultado.centroides.copy()
        if perfiles.shape[1] > max_variables:
            variables = perfiles.var(axis=0).nlargest(max_variables).index
            perfiles = perfiles.loc[:, variables]
        figura, eje = plt.subplots(
            figsize=(max(9, perfiles.shape[1] * 0.45), max(3, len(perfiles) * 0.8))
        )
        sns.heatmap(perfiles, cmap="vlag", center=0, annot=True, fmt=".2f", ax=eje)
        eje.set_title(f"Perfiles de {resultado.algoritmo}")
        eje.set_xlabel("Variable transformada")
        eje.set_ylabel("Cluster")
        figura.tight_layout()
        return figura

    @staticmethod
    def dendrograma(resultado: ResultadoCluster, max_etiquetas: int = 80):
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
        figura, eje = plt.subplots(figsize=(12, 6))
        dendrogram(
            resultado.matriz_vinculacion,
            labels=etiquetas,
            no_labels=not mostrar_etiquetas,
            leaf_rotation=90,
            ax=eje,
        )
        eje.set_title(f"Dendrograma {resultado.algoritmo}")
        eje.set_xlabel("Observaciones")
        eje.set_ylabel("Distancia")
        figura.tight_layout()
        return figura

    @staticmethod
    def curva_evaluacion(tabla: pd.DataFrame, algoritmo: str):
        """Grafica silhouette y, cuando existe, la inercia del benchmark."""
        figura, eje = plt.subplots(figsize=(9, 5))
        if "metodo" in tabla.columns:
            sns.lineplot(
                data=tabla,
                x="k",
                y="silhouette",
                hue="metodo",
                marker="o",
                ax=eje,
            )
        else:
            sns.lineplot(
                data=tabla,
                x="k",
                y="silhouette",
                marker="o",
                color="#1A3C2B",
                ax=eje,
            )
        eje.set_title(f"Evaluación de {algoritmo}")
        eje.set_ylabel("Silhouette")
        eje.set_xlabel("Número de clusters")
        figura.tight_layout()
        return figura

    @staticmethod
    def proyeccion(resultado: ResultadoProyeccion, etiquetas=None):
        """Grafica una proyección t-SNE o UMAP, opcionalmente coloreada."""
        datos = resultado.coordenadas.copy()
        figura, eje = plt.subplots(figsize=(9, 6))
        if etiquetas is None:
            eje.scatter(datos.iloc[:, 0], datos.iloc[:, 1], alpha=0.75, s=42)
        else:
            datos["grupo"] = pd.Series(etiquetas, index=datos.index).astype(str)
            sns.scatterplot(
                data=datos,
                x=datos.columns[0],
                y=datos.columns[1],
                hue="grupo",
                palette="tab10",
                s=45,
                alpha=0.78,
                ax=eje,
            )
        eje.set_title(f"Proyección {resultado.algoritmo}")
        figura.tight_layout()
        return figura

    @staticmethod
    def _validar_acp_2d(resultado: ResultadoACP) -> None:
        """Garantiza que la visualización tenga dos componentes disponibles."""
        if resultado.coordenadas.shape[1] < 2:
            raise ValueError("La visualización requiere al menos dos componentes.")


class VisualizadorSupervisado:
    """Visualizaciones para flujos supervisados."""

    @staticmethod
    def matriz_confusion(resultado: ResultadoClasificacion):
        """Dibuja una matriz de confusión con formato de tabla."""
        figura, eje = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            resultado.matriz_confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=eje,
            linewidths=0.4,
            linecolor="white",
        )
        eje.set_title(f"Matriz de confusión — {resultado.algoritmo}")
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Real")
        figura.tight_layout()
        return figura

    @staticmethod
    def metricas_barras(resultado: ResultadoClasificacion):
        """Compara precisión, recall y F1 por clase."""
        metricas = resultado.metricas.get("precision", {}).get("por_clase", {})
        if not metricas:
            return None

        etiquetas = list(metricas.keys())
        precision = [valor["precision"] for valor in metricas.values()]
        recall = [valor["recall"] for valor in metricas.values()]
        f1 = [valor["f1"] for valor in metricas.values()]

        frame = pd.DataFrame(
            {"Precision": precision, "Recall": recall, "F1": f1}, index=etiquetas
        )
        figura, eje = plt.subplots(figsize=(8, 4))
        frame.plot(kind="bar", ax=eje)
        eje.set_title("Métricas por clase")
        eje.set_xlabel("Clase")
        eje.set_ylabel("Valor")
        eje.set_ylim(0, 1.05)
        eje.legend(loc="lower right")
        figura.tight_layout()
        return figura

    @staticmethod
    def distribucion_estratificada(resultado: ResultadoClasificacion):
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
        figura, eje = plt.subplots(figsize=(8, 4))
        marco.plot(kind="bar", ax=eje)
        eje.set_title("Distribución estratificada en test")
        eje.set_xlabel("Clase")
        eje.set_ylabel("Frecuencia")
        eje.legend(loc="best")
        figura.tight_layout()
        return figura
