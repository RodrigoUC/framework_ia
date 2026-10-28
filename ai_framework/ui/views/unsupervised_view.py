"""Vistas de exploración de datos y aprendizaje no supervisado."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_framework.data.eda import EDA
from ai_framework.models.unsupervised import DependenciaOpcionalError
from ai_framework.ui.controllers.unsupervised_controller import (
    _columnas_numericas,
    comparar_metodos_hac,
    ejecutar_acp,
    ejecutar_hac,
    ejecutar_kmeans,
    ejecutar_kmedoids,
    ejecutar_tsne,
    ejecutar_umap,
    evaluar_kmeans_k,
)
from ai_framework.ui.models.session_state import (
    _firma,
    _guardar_resultado,
    _recuperar_resultado,
)
from ai_framework.ui.views.theme import _mostrar_figura
from ai_framework.ui.views.controls_view import _seleccionar_entero
from ai_framework.ui.views.data_view import _resumen_dataset_modulo
from ai_framework.utils.visualization import VisualizadorNoSupervisado


def _render_eda(datos: pd.DataFrame) -> None:
    """Presenta gráficos exploratorios generados por EDA."""
    _resumen_dataset_modulo(datos)
    eda = EDA(dataframe=datos)
    (
        resumen,
        histogramas,
        boxplots,
        dispersion,
        correlacion,
        outliers,
        frecuencias,
    ) = st.tabs(
        [
            "Resumen",
            "Histogramas",
            "Boxplots",
            "Dispersión",
            "Correlación",
            "Outliers",
            "Frecuencias",
        ]
    )

    with resumen:
        for nombre, tabla in eda.distribucion_variables().items():
            if not tabla.empty:
                st.markdown(f"**Variables {nombre}**")
                st.dataframe(tabla, width="stretch")

    numericas = _columnas_numericas(datos)
    with histogramas:
        seleccion = st.multiselect(
            "Variables numéricas",
            numericas,
            default=numericas[:3],
            key="eda_histogramas",
        )
        if seleccion:
            figura = eda.histogramas(seleccion, mostrar=False)
            _mostrar_figura(figura)

    with boxplots:
        seleccion = st.multiselect(
            "Variables numéricas",
            numericas,
            default=numericas[:3],
            key="eda_boxplots",
        )
        if seleccion:
            figura = eda.boxplots(seleccion, mostrar=False)
            _mostrar_figura(figura)

    with dispersion:
        seleccion = st.multiselect(
            "Variables numéricas",
            numericas,
            default=numericas[:4],
            key="eda_dispersion",
        )
        if len(seleccion) >= 2 and st.button("Generar dispersión"):
            grafico = eda.scatterplots(seleccion, mostrar=False)
            _mostrar_figura(grafico)

    with correlacion:
        seleccion = st.multiselect(
            "Variables numéricas",
            numericas,
            default=numericas[:8],
            key="eda_correlacion",
        )
        metodo = st.selectbox(
            "Método", ["pearson", "spearman", "kendall"], key="eda_metodo"
        )
        if len(seleccion) >= 2:
            figura, matriz = eda.mapa_calor(
                seleccion, metodo=metodo, mostrar=False
            )
            _mostrar_figura(figura)
            st.dataframe(matriz, width="stretch")

    with outliers:
        factor = st.slider("Factor IQR", 1.0, 3.0, 1.5, 0.1)
        if numericas:
            resultado = eda.detectar_outliers(numericas, factor_iqr=factor)
            st.dataframe(resultado["resumen"], width="stretch")
            st.metric(
                "Filas con algún valor atípico",
                len(resultado["filas_con_algun_outlier"]),
            )

    with frecuencias:
        seleccion = st.multiselect(
            "Variables",
            datos.columns.tolist(),
            default=datos.columns.tolist()[:3],
            key="eda_frecuencias",
        )
        for columna, tabla in eda.frecuencias(seleccion).items():
            with st.expander(f"Frecuencias de {columna}"):
                st.dataframe(tabla, width="stretch")


def _render_acp(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta y muestra el análisis de componentes principales."""
    st.markdown("### Análisis de componentes principales")
    if len(datos) < 2 or len(configuracion["features"]) < 2:
        st.warning(
            "Seleccione al menos dos variables y dos filas para ejecutar el ACP."
        )
        return
    maximo = min(10, len(datos), max(2, len(configuracion["features"])))
    componentes = _seleccionar_entero(
        "Componentes", 2, maximo, 2, key="acp_componentes"
    )
    firma = _firma(configuracion, componentes)
    if st.button("Ejecutar ACP", type="primary"):
        try:
            resultado = ejecutar_acp(datos, configuracion, n_componentes=componentes)
            _guardar_resultado("resultado_acp", firma, resultado)
        except Exception as exc:
            st.error(f"No fue posible ejecutar el ACP: {exc}")

    resultado = _recuperar_resultado("resultado_acp", firma)
    if resultado is None:
        return
    st.dataframe(
        pd.concat(
            [resultado.varianza_explicada, resultado.varianza_acumulada], axis=1
        ),
        width="stretch",
    )
    _mostrar_figura(VisualizadorNoSupervisado.varianza_acp(resultado))
    plano, circulo, sobreposicion = st.tabs(
        ["Plano principal", "Círculo de correlación", "Sobreposición"]
    )
    with plano:
        _mostrar_figura(VisualizadorNoSupervisado.plano_acp(resultado))
    with circulo:
        _mostrar_figura(
            VisualizadorNoSupervisado.circulo_correlacion(resultado)
        )
    with sobreposicion:
        _mostrar_figura(VisualizadorNoSupervisado.sobreposicion_acp(resultado))


def _render_particional(
    datos: pd.DataFrame, configuracion: dict, algoritmo: str
) -> None:
    """Renderiza una técnica particional con el flujo común de resultados."""
    st.markdown(f"### {algoritmo}")
    if len(datos) < 3 or not configuracion["features"]:
        st.warning("Se requieren variables y al menos tres filas.")
        return
    limite = min(12, len(datos) - 1)
    clusters = _seleccionar_entero(
        "Número de clusters", 2, limite, 3, key=f"{algoritmo.lower()}_clusters"
    )
    firma = _firma(configuracion, algoritmo, clusters)
    clave_resultado = f"resultado_{algoritmo.lower().replace('-', '_')}"
    if st.button(f"Ejecutar {algoritmo}", type="primary"):
        try:
            if algoritmo == "K-Means":
                resultado = ejecutar_kmeans(datos, configuracion, clusters)
            else:
                resultado = ejecutar_kmedoids(datos, configuracion, clusters)
            _guardar_resultado(clave_resultado, firma, resultado)
        except Exception as exc:
            st.error(f"No fue posible agrupar el dataset: {exc}")

    resultado = _recuperar_resultado(clave_resultado, firma)
    if resultado is not None:
        c1, c2 = st.columns(2)
        c1.metric(
            "Silhouette",
            f"{resultado.silhouette:.4f}"
            if resultado.silhouette is not None
            else "No disponible",
        )
        c2.metric(
            "Inercia",
            f"{resultado.inercia:.4f}"
            if resultado.inercia is not None
            else "No aplica",
        )
        _mostrar_figura(VisualizadorNoSupervisado.clusters(resultado))
        _mostrar_figura(VisualizadorNoSupervisado.perfiles_cluster(resultado))
        salida = datos.loc[resultado.etiquetas.index].copy()
        salida["cluster"] = resultado.etiquetas
        st.download_button(
            "Descargar asignaciones",
            salida.to_csv(index=False).encode("utf-8"),
            file_name="clusters.csv",
            mime="text/csv",
        )

    if algoritmo == "K-Means":
        firma_benchmark = _firma(configuracion, "benchmark_kmeans")
        if st.button("Evaluar valores de k", key="btn_evaluar_kmeans"):
            try:
                limite_k = min(10, len(datos) - 1)
                tabla = evaluar_kmeans_k(datos, configuracion, k_min=2, k_max=limite_k)
                _guardar_resultado("benchmark_kmeans", firma_benchmark, tabla)
            except Exception as exc:
                st.error(f"No fue posible evaluar k: {exc}")
        tabla = _recuperar_resultado("benchmark_kmeans", firma_benchmark)
        if tabla is not None:
            st.dataframe(tabla, width="stretch")
            _mostrar_figura(
                VisualizadorNoSupervisado.curva_evaluacion(tabla, "K-Means")
            )

    if algoritmo == "K-Means":
        with st.expander("Proyecciones para visualizar los clusters"):
            etiquetas = (
                resultado.etiquetas if resultado is not None else None
            )
            tab_tsne, tab_umap = st.tabs(["t-SNE", "UMAP"])
            with tab_tsne:
                _render_tsne(
                    datos, configuracion, etiquetas=etiquetas, firma_cluster=firma
                )
            with tab_umap:
                _render_umap(
                    datos, configuracion, etiquetas=etiquetas, firma_cluster=firma
                )


def _render_hac(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta clustering jerárquico y presenta el dendrograma."""
    st.markdown("### Clustering jerárquico aglomerativo")
    if len(datos) < 3 or not configuracion["features"]:
        st.warning("Se requieren variables y al menos tres filas.")
        return
    metodo = st.selectbox(
        "Vinculación", ["ward", "average", "complete", "single"]
    )
    limite = min(12, len(datos) - 1)
    clusters = _seleccionar_entero(
        "Número de clusters", 2, limite, 3, key="hac_clusters"
    )
    filas_hac = min(configuracion["filas"], 800)
    firma = _firma(configuracion, metodo, clusters, filas_hac)
    if st.button("Ejecutar HAC", type="primary"):
        try:
            resultado = ejecutar_hac(
                datos,
                configuracion,
                clusters=clusters,
                metodo=metodo,
                filas_hac=filas_hac,
            )
            _guardar_resultado("resultado_hac", firma, resultado)
        except Exception as exc:
            st.error(f"No fue posible ejecutar HAC: {exc}")

    resultado = _recuperar_resultado("resultado_hac", firma)
    if resultado is not None:
        st.metric(
            "Silhouette",
            f"{resultado.silhouette:.4f}"
            if resultado.silhouette is not None
            else "No disponible",
        )
        _mostrar_figura(VisualizadorNoSupervisado.dendrograma(resultado))
        _mostrar_figura(VisualizadorNoSupervisado.clusters(resultado))
        _mostrar_figura(VisualizadorNoSupervisado.perfiles_cluster(resultado))

    firma_benchmark = _firma(configuracion, "benchmark_hac", filas_hac)
    if st.button("Comparar métodos HAC", key="btn_comparar_hac"):
        try:
            limite_k = min(8, len(datos) - 1)
            tabla = comparar_metodos_hac(
                datos, configuracion, filas_hac=filas_hac, k_min=2, k_max=limite_k
            )
            _guardar_resultado("benchmark_hac", firma_benchmark, tabla)
        except Exception as exc:
            st.error(f"No fue posible comparar HAC: {exc}")

    tabla = _recuperar_resultado("benchmark_hac", firma_benchmark)
    if tabla is not None:
        st.dataframe(tabla, width="stretch")
        _mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(tabla, "HAC"))


def _render_tsne(
    datos: pd.DataFrame,
    configuracion: dict,
    *,
    etiquetas=None,
    firma_cluster=None,
) -> None:
    """Ejecuta t-SNE bajo demanda para contextualizar la solución K-Means."""
    st.markdown("### Proyección t-SNE")
    filas = min(configuracion["filas"], 2000)
    if filas < 3 or not configuracion["features"]:
        st.warning("t-SNE requiere variables y al menos tres filas.")
        return
    max_perplexity = min(50, filas - 1)
    perplexity = _seleccionar_entero(
        "Perplexity", 2, max_perplexity, 30, key="tsne_perplexity"
    )
    iteraciones = st.slider("Iteraciones", 250, 2000, 1000, 250, key="tsne_iteraciones")
    firma = _firma(configuracion, perplexity, iteraciones, filas, firma_cluster)
    clave = "resultado_kmeans_tsne"
    if st.button("Ejecutar t-SNE", type="primary"):
        try:
            resultado = ejecutar_tsne(
                datos,
                configuracion,
                filas=filas,
                perplexity=perplexity,
                iteraciones=iteraciones,
                etiquetas=etiquetas,
            )
            _guardar_resultado(clave, firma, resultado)
        except Exception as exc:
            st.error(f"No fue posible ejecutar t-SNE: {exc}")
    resultado = _recuperar_resultado(clave, firma)
    if resultado is not None:
        grupos = (
            etiquetas.reindex(resultado.coordenadas.index)
            if etiquetas is not None
            else None
        )
        _mostrar_figura(
            VisualizadorNoSupervisado.proyeccion(resultado, grupos)
        )
        st.dataframe(resultado.coordenadas, width="stretch")


def _render_umap(
    datos: pd.DataFrame,
    configuracion: dict,
    *,
    etiquetas=None,
    firma_cluster=None,
) -> None:
    """Ejecuta UMAP bajo demanda para contextualizar la solución K-Means."""
    st.markdown("### Proyección UMAP")
    filas = min(configuracion["filas"], 3000)
    if filas < 3 or not configuracion["features"]:
        st.warning("UMAP requiere variables y al menos tres filas.")
        return
    max_vecinos = min(100, filas - 1)
    vecinos = _seleccionar_entero(
        "Vecinos", 2, max_vecinos, 15, key="umap_vecinos"
    )
    distancia = st.slider("Distancia mínima", 0.0, 0.99, 0.1, 0.05, key="umap_distancia")
    firma = _firma(configuracion, vecinos, distancia, filas, firma_cluster)
    clave = "resultado_kmeans_umap"
    if st.button("Ejecutar UMAP", type="primary"):
        try:
            resultado = ejecutar_umap(
                datos,
                configuracion,
                filas=filas,
                vecinos=vecinos,
                distancia=distancia,
                etiquetas=etiquetas,
            )
            _guardar_resultado(clave, firma, resultado)
        except DependenciaOpcionalError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"No fue posible ejecutar UMAP: {exc}")
    resultado = _recuperar_resultado(clave, firma)
    if resultado is not None:
        grupos = (
            etiquetas.reindex(resultado.coordenadas.index)
            if etiquetas is not None
            else None
        )
        _mostrar_figura(
            VisualizadorNoSupervisado.proyeccion(resultado, grupos)
        )
        st.dataframe(resultado.coordenadas, width="stretch")
