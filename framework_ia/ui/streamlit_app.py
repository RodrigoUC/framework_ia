"""Aplicación Streamlit del framework de análisis de datos.

La interfaz coordina objetos de dominio, pero no implementa algoritmos. Para
analizar otro dataset basta con reemplazar un CSV local o subir uno nuevo.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from ..datos.eda import EDA
from ..datos.fuentes import CargadorCSV, ConfiguracionCSV
from ..datos.particion import ConfiguracionParticion, Particionador
from ..modelos.no_supervisado import Cluster, DependenciaOpcionalError
from ..resultados import ResultadoParticion
from ..visualizacion import (
    VisualizadorDatos,
    VisualizadorNoSupervisado,
)
from .configuracion_clasificacion import configuracion_clasificacion_actual
from .configuracion_clasificacion import (
    configurar_clasificacion as _configurar_modelos_clasificacion,
)
from .configuracion_clasificacion import (
    firma_clasificacion as _firma_clasificacion,
)
from .estado_clasificacion import (
    limpiar_resultados_clasificacion,
    sincronizar_contexto,
)
from .lab2 import (
    preparacion_segura,
    render_experimentos,
    render_modelo_individual,
    render_resultados,
)

BASE_DIR = Path(__file__).resolve().parents[2]
PALETA = {
    "fondo": "#f7f7f5",
    "superficie": "#eeeee9",
    "texto": "#242523",
    "acento": "#1a3c2b",
    "acento_alt": "#ff8c69",
    "muted": "#5f635e",
    "borde": "#a8aaa2",
}


PALETA_OSCURA = {
    "fondo": "#111713",
    "superficie": "#1a241d",
    "texto": "#f7f7f5",
    "borde": "#556158",
}


VISTAS_NO_SUPERVISADAS = {"acp", "kmeans", "kmedoids", "hac"}
VISTAS_CLASIFICADORES = {
    "clasificacion": "RF",
    "clasificacion_knn": "KNN",
    "clasificacion_dt": "DT",
    "clasificacion_xgboost": "XGBoost",
    "clasificacion_adaboost": "AdaBoost",
    "clasificacion_nb": "NR",
}
TITULOS_VISTA = {
    "datos": "Datos y preparación",
    "eda": "Exploración de datos",
    "acp": "ACP",
    "kmeans": "K-Means",
    "kmedoids": "K-Medoids",
    "hac": "Clustering jerárquico",
    "clasificacion": "Clasificación",
    "clasificacion_knn": "KNN",
    "clasificacion_dt": "Árbol de decisión",
    "clasificacion_xgboost": "XGBoost",
    "clasificacion_adaboost": "AdaBoost",
    "clasificacion_nb": "Naive Bayes",
    "clasificacion_configuracion": "Configuración de clasificación",
    "lab2_experimentos": "Comparar configuraciones",
    "lab2_resultados": "Resultados de clasificación",
    "regresion": "Regresión",
}
RUTAS_VISTA = {
    "datos": "Datos / Preparación",
    "eda": "Exploración / EDA",
    "acp": "Exploración / ACP",
    "kmeans": "Clustering / K-Means",
    "kmedoids": "Clustering / K-Medoids",
    "hac": "Clustering / HAC",
    "clasificacion": "Clasificación / Modelos disponibles",
    "clasificacion_knn": "Clasificación / KNN",
    "clasificacion_dt": "Clasificación / Árbol de decisión",
    "clasificacion_xgboost": "Clasificación / XGBoost",
    "clasificacion_adaboost": "Clasificación / AdaBoost",
    "clasificacion_nb": "Clasificación / Naive Bayes",
    "clasificacion_configuracion": "Configuración / Clasificación",
    "lab2_experimentos": "Clasificación / Comparar configuraciones",
    "lab2_resultados": "Resultados / Clasificación / Validación y prueba",
    "regresion": "Regresión / Próximamente",
}


def _aplicar_estilos_atlas() -> None:
    """Instala la capa visual compartida de la experiencia Atlas Analítico."""
    oscuro = st.context.theme.type == "dark"
    focus_color = "#9effbf" if oscuro else "#1a3c2b"
    chip_background = "#9effbf" if oscuro else "#1a3c2b"
    chip_foreground = "#111713" if oscuro else "#f7f7f5"
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap');
        @import url('https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600&display=swap');

        .stApp { font-family: 'General Sans', sans-serif; }
        :root { --atlas-focus-color: __ATLAS_FOCUS_COLOR__; }
        .stApp::selection { background: #9effbf; color: #1a3c2b; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
        code, [data-testid="stCaptionContainer"] { font-family: 'JetBrains Mono', monospace; }
        [data-testid="stSidebar"] { border-right: 1px solid color-mix(in srgb, currentColor 28%, transparent); }
        [data-testid="stSidebar"] .stButton > button { width: 100%; min-height: 2.35rem; justify-content: flex-start; border: 1px solid transparent; border-radius: 2px; font-family: 'General Sans', sans-serif; font-size: .88rem; }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] { border-left: 2px solid var(--primary-color); }
        [data-testid="stSidebar"] details { border: 1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius: 2px; background: color-mix(in srgb, currentColor 4%, transparent); }
        [data-testid="stSidebar"] summary { font-family: 'Space Grotesk', sans-serif; font-size: .9rem; }
        .atlas-nav-label { margin: 1.35rem 0 .35rem; color: color-mix(in srgb, currentColor 72%, transparent); font-family: 'JetBrains Mono', monospace; font-size: .67rem; font-weight: 500; letter-spacing: .14em; text-transform: uppercase; }
        .atlas-family { margin: .7rem 0 .15rem; color: color-mix(in srgb, currentColor 72%, transparent); font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; }
        .atlas-header { margin: .25rem 0 1.25rem; padding: 1.35rem 1.55rem; border: 1px solid color-mix(in srgb, currentColor 28%, transparent); background: transparent; }
        .atlas-breadcrumb { color: inherit; opacity: .82; font-family: 'JetBrains Mono', monospace; font-size: .68rem; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; }
        .atlas-title { margin: .45rem 0 .25rem; color: inherit; font-family: 'Space Grotesk', sans-serif; font-size: 2.15rem; font-weight: 600; letter-spacing: -.03em; }
        .atlas-subtitle { margin: 0; color: inherit; opacity: .78; font-size: .94rem; }
        [data-testid="stMetric"] { border: 1px solid color-mix(in srgb, currentColor 28%, transparent); border-radius: 2px; background: transparent; padding: .8rem .9rem; }
        [data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .07em; text-transform: uppercase; }
        [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
        .stButton > button[kind="primary"] { border-radius: 2px; }
        .stButton > button:focus-visible, input:focus-visible { outline: 2px solid var(--atlas-focus-color) !important; outline-offset: 2px; }
        [data-testid="stSidebar"] .stButton > button:focus-visible, [data-testid="stSidebar"] input:focus-visible { outline-color: #f4d35e !important; }
        [data-testid="stDataFrame"] { border: 1px solid color-mix(in srgb, currentColor 28%, transparent); }
        [data-testid="stMultiSelect"] [data-tag] { background-color: __ATLAS_CHIP_BACKGROUND__; color: __ATLAS_CHIP_FOREGROUND__; }
        [data-testid="stMultiSelect"] [data-tag] *, [data-testid="stMultiSelect"] [data-tag] svg { color: __ATLAS_CHIP_FOREGROUND__; fill: __ATLAS_CHIP_FOREGROUND__; }
        @media (max-width: 900px) { .atlas-header { padding: 1rem; } .atlas-title { font-size: 1.65rem; } }
        </style>
        """.replace("__ATLAS_FOCUS_COLOR__", focus_color)
        .replace("__ATLAS_CHIP_BACKGROUND__", chip_background)
        .replace("__ATLAS_CHIP_FOREGROUND__", chip_foreground),
        unsafe_allow_html=True,
    )


def _seleccionar_fuente() -> tuple[pd.DataFrame | None, str, str]:
    """Permite preparar una fuente y aplicarla solo al confirmar el formulario."""
    archivos = sorted((BASE_DIR / "data").glob("*.csv"))
    error_inicio = None
    if "fuente_aplicada" not in st.session_state and archivos:
        try:
            _cargar_fuente_csv(archivos[0], ConfiguracionCSV())
        except Exception as exc:  # pylint: disable=broad-except
            error_inicio = str(exc)

    with st.sidebar:
        with st.expander("Fuente de datos", expanded=False):
            if error_inicio:
                st.error(
                    "No fue posible cargar el CSV inicial. Elija otra fuente y "
                    f"pulse «Aplicar fuente». Detalle: {error_inicio}"
                )
            # Source type changes only the controls, never the applied dataset.
            origen = st.radio("Origen", ["CSV local", "Subir CSV"], key="fuente_origen")
            with st.form("form_fuente_datos"):
                separador = st.text_input("Separador", value=",", max_chars=3, key="fuente_separador")
                decimal = st.text_input("Separador decimal", value=".", max_chars=1, key="fuente_decimal")
                encoding = st.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"], key="fuente_encoding")
                usar_indice = st.checkbox("Usar primera columna como índice", key="fuente_usar_indice")
                seleccion = None
                archivo = None
                if origen == "CSV local":
                    if archivos:
                        seleccion = st.selectbox("Archivo", archivos, format_func=lambda ruta: ruta.name, key="fuente_archivo")
                    else:
                        st.warning("No hay archivos CSV en data/. Use la opción de carga.")
                else:
                    archivo = st.file_uploader("Archivo CSV", type=["csv"], key="fuente_upload")
                aplicar = st.form_submit_button("Aplicar fuente", type="primary")

            if aplicar:
                try:
                    configuracion = ConfiguracionCSV(
                        separador=separador, decimal=decimal, encoding=encoding,
                        usar_primera_columna_como_indice=usar_indice,
                    )
                    if not separador:
                        raise ValueError("El separador del CSV no puede estar vacío.")
                    if origen == "CSV local" and seleccion is not None:
                        _cargar_fuente_csv(seleccion, configuracion)
                    elif origen == "Subir CSV" and archivo is not None:
                        contenido = archivo.getvalue()
                        datos = CargadorCSV.cargar_bytes(contenido, configuracion)
                        digest = hashlib.sha256(contenido).hexdigest()
                        st.session_state["fuente_aplicada"] = (
                            datos, archivo.name, f"upload:{digest}:{configuracion}"
                        )
                    else:
                        st.warning("Seleccione un archivo CSV antes de aplicarlo.")
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"No fue posible aplicar la fuente: {exc}")
    return st.session_state.get("fuente_aplicada", (None, "", ""))


def _cargar_fuente_csv(seleccion: Path, configuracion: ConfiguracionCSV) -> None:
    """Carga un CSV local y conserva la identidad de archivo y lectura."""
    datos = CargadorCSV.cargar_ruta(seleccion, configuracion)
    identidad = f"local:{seleccion.resolve()}:{seleccion.stat().st_mtime_ns}:{configuracion}"
    st.session_state["fuente_aplicada"] = (datos, seleccion.name, identidad)


def _seleccionar_vista() -> str:
    """Renderiza la navegación multinivel y devuelve la vista activa."""
    vista = st.session_state.setdefault("vista_activa", "datos")
    if vista == "comparacion":
        vista = "lab2_resultados"
    elif vista in {"tsne", "umap"}:
        vista = "kmeans"
    st.session_state["vista_activa"] = vista

    def navegar(destino: str) -> None:
        st.session_state["vista_activa"] = destino

    def boton(etiqueta: str, destino: str, *, disabled: bool = False) -> None:
        st.button(
            etiqueta,
            key=f"nav_{destino}",
            type="primary" if vista == destino else "secondary",
            disabled=disabled,
            on_click=navegar if not disabled else None,
            args=(destino,) if not disabled else None,
        )

    with st.sidebar:
        st.markdown(
            '<p class="atlas-nav-label">Navegación</p>', unsafe_allow_html=True
        )
        boton("Datos y preparación", "datos")
        st.markdown(
            '<p class="atlas-nav-label">Exploración de datos</p>',
            unsafe_allow_html=True,
        )
        boton("EDA", "eda")
        boton("ACP", "acp")

        st.markdown(
            '<p class="atlas-nav-label">Análisis y modelos</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="atlas-nav-label">Configuración</p>',
            unsafe_allow_html=True,
        )
        boton("Configuración de clasificación", "clasificacion_configuracion")
        with st.expander(
            "Clustering", expanded=vista in {"kmeans", "kmedoids", "hac"}
        ):
            boton("K-Means", "kmeans")
            boton("K-Medoids", "kmedoids")
            boton("HAC", "hac")

        with st.expander("Clasificación", expanded=vista in {*VISTAS_CLASIFICADORES, "lab2_experimentos"}):
            st.markdown(
                '<p class="atlas-family">Modelos disponibles</p>',
                unsafe_allow_html=True,
            )
            boton("KNN", "clasificacion_knn")
            boton("Árbol de decisión", "clasificacion_dt")
            boton("Random Forest", "clasificacion")
            boton("XGBoost", "clasificacion_xgboost")
            boton("AdaBoost", "clasificacion_adaboost")
            boton("Naive Bayes", "clasificacion_nb")
            boton("Comparar configuraciones", "lab2_experimentos")

        with st.expander("Regresión", expanded=vista == "regresion"):
            st.markdown(
                '<p class="atlas-family">En preparación</p>',
                unsafe_allow_html=True,
            )
            boton("Regresión · Próximamente", "regresion")

        st.markdown(
            '<p class="atlas-nav-label">Resultados</p>', unsafe_allow_html=True
        )
        boton("Resultados de clasificación", "lab2_resultados")
    return st.session_state["vista_activa"]


def _render_encabezado_atlas(datos: pd.DataFrame, etiqueta: str, vista: str) -> None:
    """Show the current dataset and route; data health belongs to data/EDA views."""
    titulo = TITULOS_VISTA[vista]
    ruta = RUTAS_VISTA[vista]
    nombre_dataset = escape(etiqueta)
    st.markdown(
        f"""
        <section class="atlas-header">
          <div class="atlas-breadcrumb">{ruta}</div>
          <h1 class="atlas-title">{titulo}</h1>
          <p class="atlas-subtitle">Dataset activo: <strong>{nombre_dataset}</strong></p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if vista in {"datos", "eda"}:
        resumen = EDA(dataframe=datos).resumen_calidad()
        columnas = st.columns(4)
        columnas[0].metric("Filas", resumen["filas"])
        columnas[1].metric("Columnas", resumen["columnas"])
        columnas[2].metric("Nulos", resumen["nulos"])
        columnas[3].metric("Duplicados", resumen["duplicados"])


def _render_regresion() -> None:
    """Explica con claridad el estado actual del contrato de regresión."""
    st.info(
        "La capa de regresión ya está separada dentro de `modelos.supervisado`, "
        "pero sus métodos todavía son una plantilla. Esta vista estará disponible "
        "cuando se implementen regresión lineal simple y múltiple."
    )


def _render_configuracion_requerida(*, sin_features: bool = False) -> None:
    """Offer a direct path when shared classification settings need attention."""
    st.info(
        "Seleccione al menos una variable predictora en Configuración de clasificación."
        if sin_features
        else "La clasificación necesita al menos una variable objetivo y una "
        "variable predictora. Revise la configuración compartida antes de continuar."
    )
    if st.button(
        "Abrir configuración de clasificación",
        key="abrir_configuracion_clasificacion",
    ):
        st.session_state["vista_activa"] = "clasificacion_configuracion"
        st.rerun()


def _sincronizar_dataset(datos: pd.DataFrame, identidad: str) -> None:
    """Reinicia resultados únicamente cuando cambia el archivo o su lectura."""
    if st.session_state.get("dataset_identidad") == identidad:
        return
    st.session_state["dataset_identidad"] = identidad
    st.session_state["dataset_original"] = datos.copy()
    st.session_state["dataset_preparado"] = datos.copy()
    _limpiar_resultados()
    st.session_state.pop("dataset_preparacion", None)
    for clave in list(st.session_state):
        if clave.startswith(("modelo_", "clasif_", "dataset_")) and clave not in {"dataset_original", "dataset_preparado", "dataset_identidad"}:
            del st.session_state[clave]


def _limpiar_resultados() -> None:
    """Descarta resultados que ya no corresponden al dataset activo."""
    for clave in list(st.session_state):
        if (
            clave.startswith("resultado_")
            or clave.startswith("benchmark_")
            or clave.startswith("preview_")
            or clave.startswith("comparacion_")
        ):
            del st.session_state[clave]
    st.session_state.pop("particion_global", None)
    st.session_state.pop("clasificacion_contexto", None)
    st.session_state.pop("lab2_firma_experimento", None)
    st.session_state.pop("lab2_firma_modelo", None)


def _resumen_calidad(datos: pd.DataFrame) -> None:
    """Muestra indicadores básicos del dataset activo."""
    resumen = EDA(dataframe=datos).resumen_calidad()
    indicadores = list(resumen.items())
    metricas_por_fila = 3
    for inicio in range(0, len(indicadores), metricas_por_fila):
        fila = indicadores[inicio : inicio + metricas_por_fila]
        columnas = st.columns(len(fila))
        for columna, (nombre, valor) in zip(columnas, fila):
            etiqueta = nombre.replace("_", " ").title()
            columna.metric(etiqueta, valor)


def _render_dataset() -> None:
    """Permite preparar, restaurar y exportar el dataset persistente."""
    original = st.session_state["dataset_original"]
    actual = st.session_state["dataset_preparado"]
    st.markdown("### Preparación del dataset")
    _resumen_calidad(actual)
    _resumen_dataset_modulo(actual)

    st.markdown("#### Tipos de columna")
    tabla_tipos = EDA(dataframe=actual).resumen_columnas()
    tabla_col, grafico_col = st.columns([3, 2])
    with tabla_col:
        st.dataframe(tabla_tipos, width="stretch")
    with grafico_col:
        _mostrar_figura(VisualizadorDatos.tipos_columnas(tabla_tipos))

    st.markdown("#### Preparación y limpieza")
    columnas_eliminar = st.multiselect(
        "Columnas a eliminar",
        options=original.columns.tolist(),
        default=[],
        key="dataset_columnas_eliminar",
        help="Se descartan del dataset original antes de aplicar el resto de la preparación.",
    )
    c1, c2, c3 = st.columns(3)
    eliminar_duplicados = c1.checkbox("Eliminar duplicados", value=True)
    imputar_nulos = c2.checkbox("Imputar valores nulos", value=True)
    escalado = c3.selectbox(
        "Escalado permanente",
        options=["Ninguno", "Normalizar", "Estandarizar"],
    )

    accion, restaurar = st.columns(2)
    if accion.button("Aplicar preparación", type="primary"):
        try:
            eda = EDA(dataframe=original)
            preparado = eda.preparar_dataset(
                columnas_eliminar=columnas_eliminar,
                eliminar_duplicados=eliminar_duplicados,
                imputar_nulos=imputar_nulos,
                normalizar=escalado == "Normalizar",
                estandarizar=escalado == "Estandarizar",
            )
            st.session_state["dataset_preparado"] = preparado.copy()
            st.session_state["dataset_preparacion"] = {
                "imputar": imputar_nulos, "escalado": escalado,
                "eliminar_duplicados": eliminar_duplicados,
            }
            _limpiar_resultados()
            st.success("La preparación se aplicó correctamente.")
            st.rerun()
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible preparar el dataset: {exc}")

    if restaurar.button("Restaurar dataset original"):
        st.session_state["dataset_preparado"] = original.copy()
        st.session_state.pop("dataset_preparacion", None)
        _limpiar_resultados()
        st.success("Se restauró el contenido original.")
        st.rerun()

    st.dataframe(actual, width="stretch")
    st.download_button(
        "Descargar dataset actual",
        data=actual.to_csv(index=False).encode("utf-8"),
        file_name="dataset_preparado.csv",
        mime="text/csv",
    )

    _render_particion(actual)


def _render_particion(datos: pd.DataFrame) -> None:
    """Calcula una partición train/test/validación reutilizable en el framework."""
    st.markdown("#### División en entrenamiento y prueba")
    st.caption(
        "La partición calculada aquí queda disponible para Clasificación y "
        "Regresión, que pueden reutilizarla en vez de dividir los datos de nuevo."
    )
    c1, c2 = st.columns(2)
    porcentaje_test = c1.slider(
        "Porcentaje de prueba (%)", 10, 50, 20, 5, key="particion_test_pct"
    )
    porcentaje_val = c2.slider(
        "Porcentaje de validación (%) · opcional",
        0,
        30,
        0,
        5,
        key="particion_val_pct",
    )
    porcentaje_train = 100 - porcentaje_test - porcentaje_val
    if porcentaje_train <= 0:
        st.error("La suma de prueba y validación debe ser menor a 100 %.")
        return
    resumen = f"Train: **{porcentaje_train}%** · Test: **{porcentaje_test}%**"
    if porcentaje_val:
        resumen += f" · Validación: **{porcentaje_val}%**"
    st.caption(resumen)

    columna_seleccionada = st.selectbox(
        "Columna para estratificar (opcional)",
        options=["Ninguna"] + datos.columns.tolist(),
        key="particion_columna_estrato",
    )
    semilla = st.number_input(
        "Semilla (random_state)",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1,
        key="particion_semilla",
    )

    if st.button("Calcular partición", type="primary", key="particion_calcular"):
        try:
            configuracion = ConfiguracionParticion(
                porcentaje_test=porcentaje_test / 100.0,
                porcentaje_validacion=porcentaje_val / 100.0,
                columna_estratificacion=(
                    None if columna_seleccionada == "Ninguna" else columna_seleccionada
                ),
                semilla=int(semilla),
            )
            st.session_state["particion_global"] = Particionador(configuracion).dividir(datos)
            # A new applied split invalidates even the read-only results route.
            for clave in ("resultado_modelos_clasificacion", "resultado_lab2_experimento",
                          "resultado_lab2_modelo", "preview_clasificacion", "clasificacion_contexto"):
                st.session_state.pop(clave, None)
            st.success("Partición calculada y disponible para el resto del framework.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible calcular la partición: {exc}")

    particion: ResultadoParticion | None = st.session_state.get("particion_global")
    if particion is None:
        return

    if particion.advertencia:
        st.warning(particion.advertencia)

    m1, m2, m3 = st.columns(3)
    m1.metric("Train", len(particion.train))
    m2.metric("Test", len(particion.test))
    m3.metric(
        "Validación", len(particion.validacion) if particion.validacion is not None else 0
    )

    grafico_izq, grafico_der = st.columns(2)
    with grafico_izq:
        _mostrar_figura(VisualizadorDatos.tamanos_particion(particion))
    with grafico_der:
        grafico_distribucion = VisualizadorDatos.distribucion_particion(particion)
        if grafico_distribucion is not None:
            _mostrar_figura(grafico_distribucion)

    vista_train, vista_test = st.columns(2)
    with vista_train:
        st.markdown("**Vista previa de train**")
        st.dataframe(particion.train.head(5), width="stretch")
    with vista_test:
        st.markdown("**Vista previa de test**")
        st.dataframe(particion.test.head(5), width="stretch")

    descarga_train, descarga_test, descarga_val = st.columns(3)
    descarga_train.download_button(
        "Descargar train.csv",
        particion.train.to_csv(index=False).encode("utf-8"),
        file_name="train.csv",
        mime="text/csv",
    )
    descarga_test.download_button(
        "Descargar test.csv",
        particion.test.to_csv(index=False).encode("utf-8"),
        file_name="test.csv",
        mime="text/csv",
    )
    if particion.validacion is not None:
        descarga_val.download_button(
            "Descargar validacion.csv",
            particion.validacion.to_csv(index=False).encode("utf-8"),
            file_name="validacion.csv",
            mime="text/csv",
        )


def _columnas_numericas(datos: pd.DataFrame) -> list[str]:
    """Retorna las variables numéricas del DataFrame activo."""
    return datos.select_dtypes(include="number").columns.tolist()


def _columnas_categoricas(datos: pd.DataFrame) -> list[str]:
    """Retorna las variables categóricas del DataFrame activo."""
    return datos.select_dtypes(exclude="number").columns.tolist()


def _resumen_dataset_modulo(datos: pd.DataFrame, objetivo: str | None = None) -> None:
    """Muestra el resumen reutilizable desde cada módulo."""
    eda = EDA(dataframe=datos)
    resumen = eda.resumen_calidad()
    numericas = _columnas_numericas(datos)
    categoricas = _columnas_categoricas(datos)
    outliers = 0
    if numericas:
        outliers = len(eda.detectar_outliers(numericas)["filas_con_algun_outlier"])
    lineas = [
        f"- Filas: **{resumen['filas']}**",
        f"- Columnas: **{resumen['columnas']}**",
        f"- Duplicados: **{resumen['duplicados']}**",
        f"- Nulos: **{resumen['nulos']}**",
        f"- Variables numéricas: **{len(numericas)}**",
        f"- Variables categóricas: **{len(categoricas)}**",
        f"- Outliers críticos: **{outliers}**",
    ]
    if objetivo:
        cardinalidad = datos[objetivo].nunique(dropna=True)
        lineas.append(f"- Cardinalidad de target: **{cardinalidad}**")
    with st.expander("Resumen del dataset activo"):
        for linea in lineas:
            st.markdown(linea)


def _render_eda(datos: pd.DataFrame) -> None:
    """Presenta gráficos exploratorios generados por EDA."""
    _resumen_dataset_modulo(datos)
    eda = EDA(dataframe=datos)
    resumen, histogramas, boxplots, dispersion, correlacion, outliers, frecuencias = (
        st.tabs(
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
            figura, matriz = eda.mapa_calor(seleccion, metodo=metodo, mostrar=False)
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


def _configurar_modelos(datos: pd.DataFrame) -> dict:
    """Recopila una configuración común para todos los modelos no supervisados."""
    st.markdown("### Configuración del análisis")
    incluir_categoricas = st.checkbox(
        "Codificar variables categóricas", value=False, key="modelo_categoricas"
    )
    if incluir_categoricas:
        opciones = datos.columns.tolist()
        predeterminadas = datos.columns.tolist()
    else:
        opciones = _columnas_numericas(datos)
        predeterminadas = opciones
    guardadas = st.session_state.get("modelo_features", [])
    if any(columna not in opciones for columna in guardadas):
        del st.session_state["modelo_features"]
    features = st.multiselect(
        "Variables de análisis",
        options=opciones,
        default=predeterminadas,
        key="modelo_features",
    )
    estandarizar = st.checkbox(
        "Estandarizar para modelado", value=True, key="modelo_estandarizar"
    )
    maximo = min(len(datos), 3000)
    if maximo > 20:
        filas = st.slider(
            "Máximo de filas por modelo",
            min_value=20,
            max_value=maximo,
            value=maximo,
            step=max(1, maximo // 20),
            key="modelo_filas",
        )
    else:
        filas = maximo
    return {
        "features": features,
        "incluir_categoricas": incluir_categoricas,
        "estandarizar": estandarizar,
        "filas": filas,
    }



def _muestrear(datos: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    """Limita análisis costosos de forma reproducible."""
    if len(datos) <= cantidad:
        return datos.copy()
    return datos.sample(n=cantidad, random_state=42).sort_index()


def _crear_reductor(datos: pd.DataFrame, configuracion: dict) -> Cluster:
    """Construye el modelo usado para ACP, t-SNE y UMAP desde la interfaz."""
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def _crear_cluster(datos: pd.DataFrame, configuracion: dict) -> Cluster:
    """Construye un modelo de agrupamiento sin lógica específica de Streamlit."""
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def _firma(configuracion: dict, *parametros) -> tuple:
    """Identifica cuándo un resultado sigue correspondiendo a los controles."""
    return (
        tuple(configuracion["features"]),
        configuracion["incluir_categoricas"],
        configuracion["estandarizar"],
        configuracion["filas"],
        *parametros,
    )



def _guardar_resultado(clave: str, firma: tuple, resultado) -> None:
    """Persiste un resultado entre los reruns de Streamlit."""
    st.session_state[clave] = {"firma": firma, "resultado": resultado}


def _recuperar_resultado(clave: str, firma: tuple):
    """Retorna el resultado únicamente si coincide con la configuración actual."""
    contenedor = st.session_state.get(clave)
    if contenedor and contenedor["firma"] == firma:
        return contenedor["resultado"]
    return None


def _color_rgb(color: str) -> tuple[int, int, int] | None:
    """Convierte colores hexadecimales o rgb() a canales RGB."""
    if not isinstance(color, str):
        return None
    color = color.strip()
    if color.startswith("#") and len(color) in {4, 7}:
        if len(color) == 4:
            color = "#" + "".join(channel * 2 for channel in color[1:])
        try:
            return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
        except ValueError:
            return None
    if color.startswith("rgb(") and color.endswith(")"):
        try:
            return tuple(int(float(channel.strip())) for channel in color[4:-1].split(",")[:3])
        except ValueError:
            return None
    colores_css = {
        "black": (0, 0, 0), "white": (255, 255, 255),
        "gray": (128, 128, 128), "grey": (128, 128, 128),
        "silver": (192, 192, 192), "red": (255, 0, 0),
        "green": (0, 128, 0), "blue": (0, 0, 255),
        "orange": (255, 165, 0), "yellow": (255, 255, 0),
        "purple": (128, 0, 128), "pink": (255, 192, 203),
        "brown": (165, 42, 42), "cyan": (0, 255, 255),
        "magenta": (255, 0, 255), "lime": (0, 255, 0),
        "navy": (0, 0, 128), "teal": (0, 128, 128),
        "olive": (128, 128, 0), "maroon": (128, 0, 0),
    }
    return colores_css.get(color.lower())


def _luminancia(color: str) -> float | None:
    """Calcula luminancia relativa sRGB para etiquetas sobre celdas Plotly."""
    canales = _color_rgb(color)
    if canales is None:
        return None
    lineales = [
        canal / 12.92 if canal / 255 <= 0.04045 else ((canal / 255 + 0.055) / 1.055) ** 2.4
        for canal in canales
    ]
    return 0.2126 * lineales[0] + 0.7152 * lineales[1] + 0.0722 * lineales[2]


def _color_texto_celda(color: str) -> str:
    """Elige negro o blanco para mantener contraste en cada celda del mapa."""
    luminancia = _luminancia(color)
    if luminancia is None:
        return "#172019"
    contraste_oscuro = (luminancia + 0.05) / 0.05
    contraste_claro = 1.05 / (luminancia + 0.05)
    return "#172019" if contraste_oscuro >= contraste_claro else "#ffffff"


def _color_texto_sector(color: str, predeterminado: str) -> str:
    """Elige texto AA para una sección circular con color explícito."""
    luminancia = _luminancia(color)
    if luminancia is None:
        return predeterminado
    return "#000000" if (luminancia + 0.05) / 0.05 >= 4.5 else "#ffffff"


def _ajustar_color_traza(color, oscuro: bool):
    """Eleva el contraste de colores de marcas y líneas sin perder su matiz."""
    if isinstance(color, (list, tuple)):
        return [_ajustar_color_traza(valor, oscuro) for valor in color]
    canales = _color_rgb(color)
    if canales is None:
        return color
    fondo = PALETA_OSCURA["superficie"] if oscuro else PALETA["superficie"]
    luminancia_fondo = _luminancia(fondo)

    def contraste(luminancia: float) -> float:
        menor, mayor = sorted((luminancia, luminancia_fondo))
        return (mayor + 0.05) / (menor + 0.05)

    luminancia = _luminancia(color)
    if luminancia is None or contraste(luminancia) >= 3:
        return color
    destino = (255, 255, 255) if oscuro else (0, 0, 0)
    for paso in range(1, 101):
        proporcion = paso / 100
        ajustado = tuple(
            round(canal + (meta - canal) * proporcion)
            for canal, meta in zip(canales, destino)
        )
        hexadecimal = "#" + "".join(f"{canal:02x}" for canal in ajustado)
        if contraste(_luminancia(hexadecimal)) >= 3:
            return hexadecimal
    return "#ffffff" if oscuro else "#000000"


def _texto_mapa_calor(figura: go.Figure) -> list[go.Scatter]:
    """Superpone etiquetas por celda con colores contrastantes y formato Plotly."""
    etiquetas = []
    for traza in figura.data:
        if traza.type != "heatmap" or not traza.texttemplate:
            continue
        valores = traza.z
        if isinstance(valores, dict):
            try:
                forma = tuple(int(dimension) for dimension in valores["shape"].split(","))
                arreglo = np.frombuffer(
                    base64.b64decode(valores["bdata"], validate=True),
                    dtype=np.dtype(valores["dtype"]),
                )
                if not forma or arreglo.size != math.prod(forma):
                    continue
                valores = arreglo.reshape(forma).tolist()
            except (KeyError, TypeError, ValueError):
                continue
        if valores is None or not len(valores) or not len(valores[0]):
            continue
        coloraxis = getattr(traza, "coloraxis", None)
        eje_color = getattr(figura.layout, coloraxis, None) if coloraxis else None
        escala = (eje_color.colorscale if eje_color else None) or traza.colorscale or "Viridis"
        minimo = eje_color.cmin if eje_color else traza.zmin
        maximo = eje_color.cmax if eje_color else traza.zmax
        centro = eje_color.cmid if eje_color else traza.zmid
        rango_automatico = minimo is None and maximo is None
        invertir = bool(
            (eje_color and eje_color.reversescale) or traza.reversescale
        )
        valores_validos = [
            float(valor)
            for fila in valores
            for valor in fila
            if valor is not None and math.isfinite(float(valor))
        ]
        if not valores_validos:
            continue
        minimo = minimo if minimo is not None else min(valores_validos)
        maximo = maximo if maximo is not None else max(valores_validos)
        if centro is not None and rango_automatico:
            radio = max(abs(minimo - centro), abs(maximo - centro))
            minimo, maximo = centro - radio, centro + radio
        xs = list(traza.x) if traza.x is not None else list(range(len(valores[0])))
        ys = list(traza.y) if traza.y is not None else list(range(len(valores)))
        xs_repetidos, ys_repetidos, celdas, posiciones = [], [], [], []
        if not isinstance(escala, str):
            escala = [list(punto) for punto in escala]
        for fila, y in zip(valores, ys):
            for valor, x in zip(fila, xs):
                if valor is None or not math.isfinite(float(valor)):
                    continue
                posicion = 0.5 if maximo == minimo else min(1, max(0, (float(valor) - minimo) / (maximo - minimo)))
                if centro is not None and rango_automatico and minimo < centro < maximo:
                    posicion = (
                        0.5 * (float(valor) - minimo) / (centro - minimo)
                        if float(valor) <= centro
                        else 0.5 + 0.5 * (float(valor) - centro) / (maximo - centro)
                    )
                if invertir:
                    posicion = 1 - posicion
                xs_repetidos.append(x)
                ys_repetidos.append(y)
                celdas.append(valor)
                posiciones.append(posicion)
        colores = [
            _color_texto_celda(color)
            for color in sample_colorscale(escala, posiciones)
        ]
        plantilla = re.sub(r"(%\{)z(?=[:}])", r"\1text", traza.texttemplate)
        etiquetas.append(
            go.Scatter(
                x=xs_repetidos,
                y=ys_repetidos,
                text=celdas,
                mode="text",
                texttemplate=plantilla,
                textfont={"color": colores, "size": 12},
                hoverinfo="skip",
                showlegend=False,
                meta="plotly-theme-cell-labels",
            )
        )
        traza.texttemplate = ""
    return etiquetas


def _aplicar_tema_figura(figura: go.Figure, oscuro: bool) -> None:
    """Alinea fondo, ejes, trazas y anotaciones con el tema Streamlit activo."""
    figura.update_layout(
        template="plotly_dark" if oscuro else "plotly_white",
        paper_bgcolor=PALETA_OSCURA["fondo"] if oscuro else PALETA["fondo"],
        plot_bgcolor=PALETA_OSCURA["superficie"] if oscuro else PALETA["superficie"],
        font_color=PALETA_OSCURA["texto"] if oscuro else PALETA["texto"],
        margin=dict(l=40, r=20, t=60, b=40),
    )
    color_texto = PALETA_OSCURA["texto"] if oscuro else PALETA["texto"]
    color_grid = "#46534a" if oscuro else "#d2d3cd"
    figura.update_xaxes(color=color_texto, gridcolor=color_grid, zerolinecolor=color_grid, linecolor=color_grid)
    figura.update_yaxes(color=color_texto, gridcolor=color_grid, zerolinecolor=color_grid, linecolor=color_grid)
    figura.update_annotations(font_color=color_texto, arrowcolor=color_texto)
    for forma in figura.layout.shapes or ():
        if forma.line.color:
            forma.line.color = _ajustar_color_traza(forma.line.color, oscuro)
    for traza in figura.data:
        for atributo in ("marker", "line"):
            estilo = getattr(traza, atributo, None)
            color = getattr(estilo, "color", None) if estilo is not None else None
            if color is not None:
                estilo.color = _ajustar_color_traza(color, oscuro)
            if estilo is not None and getattr(estilo, "colors", None) is not None:
                estilo.colors = _ajustar_color_traza(estilo.colors, oscuro)
        if (
            getattr(traza, "textfont", None) is not None
            and not (
                isinstance(traza.meta, str)
                and traza.meta == "plotly-theme-cell-labels"
            )
        ):
            if traza.type == "pie" and traza.marker.colors:
                traza.textfont.color = [
                    _color_texto_sector(color, color_texto)
                    for color in traza.marker.colors
                ]
            else:
                traza.textfont.color = color_texto
    for traza in _texto_mapa_calor(figura):
        figura.add_trace(traza)


def _mostrar_figura(figura: go.Figure, key: str | None = None, *, alt: str | None = None) -> None:
    """Renderiza una figura Plotly alineada con el tema claro/oscuro activo."""
    _aplicar_tema_figura(figura, st.context.theme.type == "dark")
    opciones = {"width": "stretch"}
    if key is not None:
        opciones["key"] = key
    if alt is not None:
        st.caption(alt)
    st.plotly_chart(figura, **opciones)


def _seleccionar_entero(
    etiqueta: str,
    minimo: int,
    maximo: int,
    predeterminado: int,
    *,
    key: str,
) -> int:
    """Evita controles inválidos cuando un dataset admite un único valor."""
    if minimo == maximo:
        st.caption(f"{etiqueta}: {minimo}")
        return minimo
    return st.slider(
        etiqueta,
        minimo,
        maximo,
        min(max(predeterminado, minimo), maximo),
        key=key,
    )


def _render_acp(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta y muestra el análisis de componentes principales."""
    st.markdown("### Análisis de componentes principales")
    if len(datos) < 2 or len(configuracion["features"]) < 2:
        st.warning("Seleccione al menos dos variables y dos filas para ejecutar el ACP.")
        return
    maximo = min(10, len(datos), max(2, len(configuracion["features"])))
    componentes = _seleccionar_entero(
        "Componentes", 2, maximo, 2, key="acp_componentes"
    )
    firma = _firma(configuracion, componentes)
    if st.button("Ejecutar ACP", type="primary"):
        try:
            muestra = _muestrear(datos, configuracion["filas"])
            resultado = _crear_reductor(muestra, configuracion).ACP(componentes)
            _guardar_resultado("resultado_acp", firma, resultado)
        except Exception as exc:  # pylint: disable=broad-except
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
        _mostrar_figura(VisualizadorNoSupervisado.circulo_correlacion(resultado))
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
            muestra = _muestrear(datos, configuracion["filas"])
            modelo = _crear_cluster(muestra, configuracion)
            if algoritmo == "K-Means":
                resultado = modelo.K_means(clusters)
            else:
                resultado = modelo.K_medoids(clusters)
            _guardar_resultado(clave_resultado, firma, resultado)
        except Exception as exc:  # pylint: disable=broad-except
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

    if algoritmo == "K-Means" and st.button("Evaluar valores de k"):
        try:
            muestra = _muestrear(datos, configuracion["filas"])
            tabla = _crear_cluster(muestra, configuracion).evaluar_kmeans(
                2, min(10, len(muestra) - 1)
            )
            st.session_state["benchmark_kmeans"] = tabla
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible evaluar k: {exc}")
    tabla = st.session_state.get("benchmark_kmeans")
    if tabla is not None:
        st.dataframe(tabla, width="stretch")
        _mostrar_figura(
            VisualizadorNoSupervisado.curva_evaluacion(tabla, "K-Means")
        )

    if algoritmo == "K-Means":
        with st.expander("Proyecciones para visualizar los clusters"):
            st.caption(
                "t-SNE y UMAP proyectan los datos; no son algoritmos de clustering."
            )
            etiquetas = resultado.etiquetas if resultado is not None else None
            _render_tsne(
                datos, configuracion, etiquetas=etiquetas, firma_cluster=firma
            )
            _render_umap(
                datos, configuracion, etiquetas=etiquetas, firma_cluster=firma
            )


def _render_hac(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta clustering jerárquico y presenta el dendrograma."""
    st.markdown("### Clustering jerárquico aglomerativo")
    if len(datos) < 3 or not configuracion["features"]:
        st.warning("Se requieren variables y al menos tres filas.")
        return
    metodo = st.selectbox("Vinculación", ["ward", "average", "complete", "single"])
    limite = min(12, len(datos) - 1)
    clusters = _seleccionar_entero(
        "Número de clusters", 2, limite, 3, key="hac_clusters"
    )
    filas_hac = min(configuracion["filas"], 800)
    firma = _firma(configuracion, metodo, clusters, filas_hac)
    st.caption(f"El dendrograma utilizará como máximo {filas_hac} filas.")
    if st.button("Ejecutar HAC", type="primary"):
        try:
            muestra = _muestrear(datos, filas_hac)
            resultado = _crear_cluster(muestra, configuracion).HAC(
                clusters, metodo=metodo
            )
            _guardar_resultado("resultado_hac", firma, resultado)
        except Exception as exc:  # pylint: disable=broad-except
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

    if st.button("Comparar métodos HAC"):
        try:
            muestra = _muestrear(datos, filas_hac)
            tabla = _crear_cluster(muestra, configuracion).evaluar_hac(
                2, min(8, len(muestra) - 1)
            )
            st.session_state["benchmark_hac"] = tabla
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible comparar HAC: {exc}")
    tabla = st.session_state.get("benchmark_hac")
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
    iteraciones = st.slider("Iteraciones", 250, 2000, 1000, 250)
    firma = _firma(configuracion, perplexity, iteraciones, filas, firma_cluster)
    clave = "resultado_kmeans_tsne"
    if st.button("Ejecutar t-SNE", type="primary"):
        try:
            muestra = (
                datos.loc[etiquetas.index[:filas]]
                if etiquetas is not None
                else _muestrear(datos, filas)
            )
            resultado = _crear_reductor(muestra, configuracion).TSNE(
                perplexity=perplexity, max_iter=iteraciones
            )
            _guardar_resultado(clave, firma, resultado)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible ejecutar t-SNE: {exc}")
    resultado = _recuperar_resultado(clave, firma)
    if resultado is not None:
        grupos = (
            etiquetas.reindex(resultado.coordenadas.index)
            if etiquetas is not None
            else None
        )
        _mostrar_figura(VisualizadorNoSupervisado.proyeccion(resultado, grupos))
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
    distancia = st.slider("Distancia mínima", 0.0, 0.99, 0.1, 0.05)
    firma = _firma(configuracion, vecinos, distancia, filas, firma_cluster)
    clave = "resultado_kmeans_umap"
    if st.button("Ejecutar UMAP", type="primary"):
        try:
            muestra = (
                datos.loc[etiquetas.index[:filas]]
                if etiquetas is not None
                else _muestrear(datos, filas)
            )
            resultado = _crear_reductor(muestra, configuracion).UMAP(
                n_neighbors=vecinos, min_dist=distancia
            )
            _guardar_resultado(clave, firma, resultado)
        except DependenciaOpcionalError as exc:
            st.warning(str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible ejecutar UMAP: {exc}")
    resultado = _recuperar_resultado(clave, firma)
    if resultado is not None:
        grupos = (
            etiquetas.reindex(resultado.coordenadas.index)
            if etiquetas is not None
            else None
        )
        _mostrar_figura(VisualizadorNoSupervisado.proyeccion(resultado, grupos))
        st.dataframe(resultado.coordenadas, width="stretch")


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Atlas Analítico",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _aplicar_estilos_atlas()

    with st.sidebar:
        st.button(
            "Actualizar gráficos",
            key="actualizar_graficos",
            icon=":material/refresh:",
            help=(
                "Úselo después de cambiar el tema en Configuración para "
                "actualizar los colores de gráficos y estilos."
            ),
        )

    datos_cargados, etiqueta, identidad = _seleccionar_fuente()
    if datos_cargados is None:
        st.markdown("# Atlas Analítico")
        st.info("Seleccione un CSV local o suba un archivo para comenzar.")
        return
    _sincronizar_dataset(datos_cargados, identidad)
    datos = st.session_state["dataset_preparado"]
    vista = _seleccionar_vista()
    _render_encabezado_atlas(datos, etiqueta, vista)

    configuracion = None
    configuracion_clasif = None
    if vista in VISTAS_NO_SUPERVISADAS:
        configuracion = _configurar_modelos(datos)
    elif vista == "clasificacion_configuracion":
        configuracion_clasif = _configurar_modelos_clasificacion(
            datos, _mostrar_figura
        )
        if configuracion_clasif is not None:
            sincronizar_contexto(st.session_state, datos, configuracion_clasif)
            preparacion_segura(mostrar_guia=True)
    elif vista in {*VISTAS_CLASIFICADORES, "lab2_experimentos"}:
        configuracion_clasif = configuracion_clasificacion_actual(
            datos, st.session_state
        )
        if configuracion_clasif is not None:
            sincronizar_contexto(st.session_state, datos, configuracion_clasif)

    elif vista == "lab2_resultados":
        configuracion_clasif = configuracion_clasificacion_actual(datos, st.session_state)
        if configuracion_clasif is None:
            limpiar_resultados_clasificacion(st.session_state)
        else:
            sincronizar_contexto(st.session_state, datos, configuracion_clasif)
            firma_legacy = _firma_clasificacion(
                configuracion_clasif,
                st.session_state.get("dataset_identidad"),
                "modelos_entrenados",
            )
            legacy = st.session_state.get("resultado_modelos_clasificacion", {}).get(
                firma_legacy, {}
            )
            for codigo, entrada in legacy.items():
                if codigo in {"RF", "NR"} and isinstance(entrada, dict):
                    st.session_state.setdefault(
                        f"resultado_lab2_modelo_{codigo}", entrada.get("resultado")
                    )

    if vista == "datos":
        _render_dataset()
    elif vista == "eda":
        _render_eda(datos)
    elif vista == "clasificacion_configuracion":
        if configuracion_clasif is None:
            st.warning(
                "El dataset necesita al menos dos columnas para configurar "
                "clasificación."
            )
    elif vista == "acp":
        assert configuracion is not None
        _render_acp(datos, configuracion)
    elif vista in VISTAS_CLASIFICADORES:
        if configuracion_clasif is None:
            _render_configuracion_requerida()
        elif not configuracion_clasif["features"]:
            _render_configuracion_requerida(sin_features=True)
        else:
            render_modelo_individual(
                datos, configuracion_clasif, _mostrar_figura,
                algoritmo=VISTAS_CLASIFICADORES[vista],
            )
    elif vista == "lab2_experimentos":
        if configuracion_clasif is None:
            _render_configuracion_requerida()
        elif not configuracion_clasif["features"]:
            _render_configuracion_requerida(sin_features=True)
        else:
            render_experimentos(datos, configuracion_clasif)
    elif vista == "lab2_resultados":
        if configuracion_clasif is None:
            st.info(
                "Los resultados no están disponibles: el dataset activo necesita "
                "al menos una columna objetivo y una columna predictora."
            )
        else:
            render_resultados(_mostrar_figura)
    elif vista == "kmeans":
        assert configuracion is not None
        _render_particional(datos, configuracion, "K-Means")
    elif vista == "kmedoids":
        assert configuracion is not None
        _render_particional(datos, configuracion, "K-Medoids")
    elif vista == "hac":
        assert configuracion is not None
        _render_hac(datos, configuracion)
    else:
        _render_regresion()


if __name__ == "__main__":
    main()
