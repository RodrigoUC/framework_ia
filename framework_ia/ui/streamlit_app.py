"""Aplicación Streamlit del framework de análisis de datos.

La interfaz coordina objetos de dominio, pero no implementa algoritmos. Para
analizar otro dataset basta con reemplazar un CSV local o subir uno nuevo.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ..datos.eda import EDA
from ..datos.fuentes import CargadorCSV, ConfiguracionCSV
from ..modelos.clasificacion import Clasificacion
from ..modelos.cluster import Cluster
from ..modelos.reduccion_dimensional import DependenciaOpcionalError, ReduccionDimensional
from ..visualizacion import (
    VisualizadorNoSupervisado,
    VisualizadorSupervisado,
)


BASE_DIR = Path(__file__).resolve().parents[2]
PALETA = {
    "fondo": "#f7f9fc",
    "superficie": "#eef2f7",
    "texto": "#1f2a37",
    "acento": "#3b82f6",
    "acento_alt": "#0ea5e9",
    "muted": "#6b7280",
    "borde": "#cbd5e1",
}


PALETA_OSCURA = {
    "fondo": "#0f172a",
    "superficie": "#162033",
    "texto": "#f8fafc",
    "borde": "#334155",
}


def _configuracion_csv() -> ConfiguracionCSV:
    """Recopila las opciones necesarias para interpretar diferentes CSV."""
    separador = st.sidebar.text_input("Separador", value=",", max_chars=3)
    decimal = st.sidebar.text_input("Separador decimal", value=".", max_chars=1)
    encoding = st.sidebar.selectbox(
        "Codificación", options=["utf-8", "latin-1", "cp1252"]
    )
    usar_indice = st.sidebar.checkbox(
        "Usar primera columna como índice", value=False
    )
    if not separador:
        raise ValueError("El separador del CSV no puede estar vacío.")
    return ConfiguracionCSV(
        separador=separador,
        decimal=decimal,
        encoding=encoding,
        usar_primera_columna_como_indice=usar_indice,
    )


def _seleccionar_fuente() -> tuple[pd.DataFrame | None, str, str]:
    """Carga un CSV local o subido y construye una identidad estable."""
    st.sidebar.markdown("## Fuente de datos")
    origen = st.sidebar.radio("Origen", options=["CSV local", "Subir CSV"])
    try:
        configuracion = _configuracion_csv()
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return None, "", ""

    if origen == "CSV local":
        archivos = sorted((BASE_DIR / "data").glob("*.csv"))
        if not archivos:
            st.sidebar.warning(
                "No hay archivos CSV en data/. Use la opción de carga."
            )
            return None, "", ""
        seleccion = st.sidebar.selectbox(
            "Archivo", options=archivos, format_func=lambda ruta: ruta.name
        )
        try:
            datos = CargadorCSV.cargar_ruta(seleccion, configuracion)
        except Exception as exc:  # pylint: disable=broad-except
            st.sidebar.error(f"No fue posible cargar el CSV: {exc}")
            return None, "", ""
        identidad = (
            f"local:{seleccion.resolve()}:{seleccion.stat().st_mtime_ns}:"
            f"{configuracion}"
        )
        return datos, seleccion.name, identidad

    archivo = st.sidebar.file_uploader("Archivo CSV", type=["csv"])
    if archivo is None:
        return None, "", ""
    contenido = archivo.getvalue()
    try:
        datos = CargadorCSV.cargar_bytes(contenido, configuracion)
    except Exception as exc:  # pylint: disable=broad-except
        st.sidebar.error(f"No fue posible cargar el CSV: {exc}")
        return None, "", ""
    digest = hashlib.sha256(contenido).hexdigest()
    return datos, archivo.name, f"upload:{digest}:{configuracion}"


def _sincronizar_dataset(datos: pd.DataFrame, identidad: str) -> None:
    """Reinicia resultados únicamente cuando cambia el archivo o su lectura."""
    if st.session_state.get("dataset_identidad") == identidad:
        return
    st.session_state["dataset_identidad"] = identidad
    st.session_state["dataset_original"] = datos.copy()
    st.session_state["dataset_preparado"] = datos.copy()
    _limpiar_resultados()
    for clave in list(st.session_state):
        if clave.startswith("modelo_"):
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


def _resumen_calidad(datos: pd.DataFrame) -> None:
    """Muestra indicadores básicos del dataset activo."""
    resumen = EDA(dataframe=datos).resumen_calidad()
    columnas = st.columns(len(resumen))
    for columna, (nombre, valor) in zip(columnas, resumen.items()):
        etiqueta = nombre.replace("_", " ").title()
        columna.metric(etiqueta, valor)


def _render_dataset() -> None:
    """Permite preparar, restaurar y exportar el dataset persistente."""
    original = st.session_state["dataset_original"]
    actual = st.session_state["dataset_preparado"]
    st.markdown("### Preparación del dataset")
    _resumen_calidad(actual)
    _resumen_dataset_modulo(actual)

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
                eliminar_duplicados=eliminar_duplicados,
                imputar_nulos=imputar_nulos,
                normalizar=escalado == "Normalizar",
                estandarizar=escalado == "Estandarizar",
            )
            st.session_state["dataset_preparado"] = preparado.copy()
            _limpiar_resultados()
            st.success("La preparación se aplicó correctamente.")
            st.rerun()
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible preparar el dataset: {exc}")

    if restaurar.button("Restaurar dataset original"):
        st.session_state["dataset_preparado"] = original.copy()
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
    """Presenta las funciones de EDA disponibles en la clase DataFrame."""
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
            figura, _ = eda.histogramas(seleccion, mostrar=False)
            _mostrar_figura(figura)

    with boxplots:
        seleccion = st.multiselect(
            "Variables numéricas",
            numericas,
            default=numericas[:3],
            key="eda_boxplots",
        )
        if seleccion:
            figura, _ = eda.boxplots(seleccion, mostrar=False)
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
            _mostrar_figura(grafico.figure)

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
            matriz = eda.mapa_calor(seleccion, metodo=metodo, mostrar=False)
            figura = plt.gcf()
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
    st.sidebar.markdown("## Modelos no supervisados")
    incluir_categoricas = st.sidebar.checkbox(
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
    features = st.sidebar.multiselect(
        "Variables de análisis",
        options=opciones,
        default=predeterminadas,
        key="modelo_features",
    )
    estandarizar = st.sidebar.checkbox(
        "Estandarizar para modelado", value=True, key="modelo_estandarizar"
    )
    maximo = min(len(datos), 3000)
    if maximo > 20:
        filas = st.sidebar.slider(
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


def _configurar_modelos_clasificacion(datos: pd.DataFrame) -> dict:
    """Recopila configuración de target, split y preprocesamiento."""
    st.sidebar.markdown("## Modelos supervisados")
    objetivo = st.sidebar.selectbox(
        "Variable objetivo (target)",
        options=datos.columns.tolist(),
        index=max(0, len(datos.columns) - 1),
        key="clasif_target",
    )
    opcional_features = [columna for columna in datos.columns if columna != objetivo]
    caracteristicas_guardadas = st.session_state.get("clasif_features", opcional_features)
    if any(columna not in opcional_features for columna in caracteristicas_guardadas):
        st.session_state["clasif_features"] = opcional_features
    caracteristicas = st.sidebar.multiselect(
        "Variables predictoras",
        options=opcional_features,
        default=opcional_features,
        key="clasif_features",
    )
    if not caracteristicas:
        st.sidebar.warning("Seleccione al menos una feature antes de entrenar.")

    st.sidebar.markdown("### Partición y reproducibilidad")
    test_size = st.sidebar.slider(
        "Tamaño de prueba (%)",
        min_value=10,
        max_value=60,
        value=25,
        step=5,
        key="clasif_test_size",
    ) / 100.0
    estado_aleatorio = st.sidebar.number_input(
        "Random state",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1,
        key="clasif_random_state",
    )
    estratificar = st.sidebar.checkbox("Estratificar por target", value=True, key="clasif_strat")

    st.sidebar.markdown("### Preprocesamiento")
    incluir_categoricas = st.sidebar.checkbox(
        "Codificar categóricas",
        value=False,
        key="clasif_incluir_categoricas",
    )
    imputar = st.sidebar.checkbox("Imputar nulos", value=True, key="clasif_imputar")
    estandarizar = st.sidebar.checkbox("Escalar (solo numéricas)", value=True, key="clasif_escalar")
    return {
        "target": objetivo,
        "features": caracteristicas,
        "test_size": test_size,
        "random_state": int(estado_aleatorio),
        "estratificar": estratificar,
        "incluir_categoricas": incluir_categoricas,
        "imputar": imputar,
        "estandarizar": estandarizar,
    }


def _muestrear(datos: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    """Limita análisis costosos de forma reproducible."""
    if len(datos) <= cantidad:
        return datos.copy()
    return datos.sample(n=cantidad, random_state=42).sort_index()


def _crear_reductor(datos: pd.DataFrame, configuracion: dict) -> ReduccionDimensional:
    """Construye un reductor a partir de la configuración de la interfaz."""
    return ReduccionDimensional(
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


def _firma_clasificacion(configuracion: dict, *parametros) -> tuple:
    """Firma específica para resultados supervisados."""
    return (
        configuracion["target"],
        tuple(configuracion["features"]),
        configuracion["test_size"],
        configuracion["random_state"],
        configuracion["estratificar"],
        configuracion["incluir_categoricas"],
        configuracion["imputar"],
        configuracion["estandarizar"],
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


def _aplicar_tema_oscuro_a_figura(figura) -> None:
    """Alinea las figuras de Matplotlib con la apariencia oscura de la app."""
    if st.context.theme.type != "dark":
        return

    fondo = PALETA_OSCURA["fondo"]
    superficie = PALETA_OSCURA["superficie"]
    texto = PALETA_OSCURA["texto"]
    borde = PALETA_OSCURA["borde"]
    figura.patch.set_facecolor(fondo)
    figura.patch.set_edgecolor(fondo)
    for eje in figura.get_axes():
        eje.set_facecolor(superficie)
        eje.tick_params(colors=texto)
        eje.xaxis.label.set_color(texto)
        eje.yaxis.label.set_color(texto)
        eje.title.set_color(texto)
        for borde_eje in eje.spines.values():
            borde_eje.set_color(borde)
        for etiqueta in (*eje.get_xticklabels(), *eje.get_yticklabels(), *eje.texts):
            etiqueta.set_color(texto)
        leyenda = eje.get_legend()
        if leyenda is not None:
            leyenda.get_frame().set_facecolor(superficie)
            leyenda.get_frame().set_edgecolor(borde)
            for etiqueta in leyenda.get_texts():
                etiqueta.set_color(texto)


def _mostrar_figura(figura) -> None:
    """Renderiza una figura con el tema activo y después la libera."""
    _aplicar_tema_oscuro_a_figura(figura)
    st.pyplot(figura)
    plt.close(figura)


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


def _mostrar_metricas_clasificacion(resultado) -> None:
    """Renderiza bloques de métricas para la clasificación."""
    metricas = resultado.metricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metricas['accuracy']:.4f}")
    c2.metric("Precision", f"{metricas['precision']['global']:.4f}")
    c3.metric("Recall", f"{metricas['recall']['global']:.4f}")
    c4.metric("F1", f"{metricas['f1']['global']:.4f}")


def _valor_metrica(resultado, nombre: str) -> float:
    """Extrae un valor estable de la métrica seleccionada."""
    metricas = resultado.metricas
    if nombre == "accuracy":
        return metricas["accuracy"]
    if nombre == "precision_global":
        return metricas["precision"]["global"]
    if nombre == "precision_macro":
        return metricas["precision"]["macro"]
    if nombre == "recall_global":
        return metricas["recall"]["global"]
    if nombre == "recall_macro":
        return metricas["recall"]["macro"]
    if nombre == "f1_global":
        return metricas["f1"]["global"]
    if nombre == "f1_macro":
        return metricas["f1"]["macro"]
    raise ValueError(f"Métrica no soportada: {nombre}")


def _render_clasificacion(datos: pd.DataFrame, configuracion: dict) -> None:
    """Flujo de configuración, previsualización y ejecución de clasificación."""
    _resumen_dataset_modulo(datos, configuracion["target"])
    st.markdown("### Clasificación supervisada")
    if not configuracion["features"]:
        st.warning("Seleccione al menos una feature para entrenar.")
        return

    objetivo = configuracion["target"]
    st.markdown(
        f"**Target:** `{objetivo}`  \n**Features:** {', '.join(configuracion['features'])}"
    )
    algoritmo = st.selectbox("Algoritmo", ["RF", "NR"], index=0, key="clasif_algoritmo")
    if algoritmo == "RF":
        with st.expander("Parámetros de Random Forest"):
            n_estimators = st.slider("N estimadores", 10, 500, 200, 10, key="clasif_rf_n")
            max_depth = st.slider("Profundidad máxima", 1, 30, 8, 1, key="clasif_rf_depth")
            min_samples_split = st.slider(
                "Min samples split",
                2,
                20,
                2,
                1,
                key="clasif_rf_split",
            )
        rf_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
        }
    else:
        rf_kwargs = {}

    firma_preview = _firma_clasificacion(configuracion, "preview")
    if st.button("Previsualizar partición", type="secondary"):
        try:
            clasificador = Clasificacion(
                dataframe=datos,
                target=objetivo,
                features=configuracion["features"],
            )
            preview = clasificador.previsualizar_particion(
                test_size=configuracion["test_size"],
                random_state=configuracion["random_state"],
                stratify=configuracion["estratificar"],
                incluir_categoricas=configuracion["incluir_categoricas"],
                imputar=configuracion["imputar"],
                estandarizar=configuracion["estandarizar"],
            )
            _guardar_resultado("preview_clasificacion", firma_preview, preview)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible previsualizar: {exc}")
    preview = _recuperar_resultado("preview_clasificacion", firma_preview)

    if preview is not None:
        st.markdown("#### Vista previa de partición")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train", preview["tam_train"])
        c2.metric("Test", preview["tam_test"])
        c3.metric("Duplicados (sel.)", preview["duplicados"])
        c4.metric("Nulos (sel.)", preview["nulos"])
        st.caption(f"Nulos % en selección: {preview['nulos_porcentaje']:.2f}%")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribución Train**")
            st.dataframe(
                pd.DataFrame.from_dict(preview["distribucion_train"], orient="index", columns=["frecuencia"]),
                width="stretch",
            )
        with c2:
            st.markdown("**Distribución Test**")
            st.dataframe(
                pd.DataFrame.from_dict(preview["distribucion_test"], orient="index", columns=["frecuencia"]),
                width="stretch",
            )
        st.markdown("**Head del split train (bruto)**")
        st.dataframe(preview["head"], width="stretch")
        st.markdown("**Tail del split test (bruto)**")
        st.dataframe(preview["tail"], width="stretch")

    if algoritmo == "RF":
        firma_entrenar = _firma_clasificacion(configuracion, "RF", tuple(sorted(rf_kwargs.items())))
    else:
        firma_entrenar = _firma_clasificacion(configuracion, "NR")
    if st.button("Entrenar modelo", type="primary"):
        try:
            clasificador = Clasificacion(
                dataframe=datos,
                target=objetivo,
                features=configuracion["features"],
            )
            if algoritmo == "RF":
                resultado = clasificador.RF(
                    test_size=configuracion["test_size"],
                    random_state=configuracion["random_state"],
                    stratify=configuracion["estratificar"],
                    incluir_categoricas=configuracion["incluir_categoricas"],
                    imputar=configuracion["imputar"],
                    estandarizar=configuracion["estandarizar"],
                    **rf_kwargs,
                )
            else:
                resultado = clasificador.NR(
                    test_size=configuracion["test_size"],
                    random_state=configuracion["random_state"],
                    stratify=configuracion["estratificar"],
                    incluir_categoricas=configuracion["incluir_categoricas"],
                    imputar=configuracion["imputar"],
                    estandarizar=configuracion["estandarizar"],
                )
            _guardar_resultado("resultado_clasificacion", firma_entrenar, resultado)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible entrenar el modelo: {exc}")

    resultado = _recuperar_resultado("resultado_clasificacion", firma_entrenar)
    if resultado is None:
        return

    st.markdown("#### Resultado del entrenamiento")
    _mostrar_metricas_clasificacion(resultado)
    _mostrar_figura(VisualizadorSupervisado.matriz_confusion(resultado))
    barras = VisualizadorSupervisado.metricas_barras(resultado)
    if barras is not None:
        _mostrar_figura(barras)
    distribucion = VisualizadorSupervisado.distribucion_estratificada(resultado)
    if distribucion is not None:
        _mostrar_figura(distribucion)

    errores = pd.DataFrame(
        {"real": resultado.y_true, "prediccion": resultado.y_pred}
    ).reset_index(drop=True)
    errores["error"] = errores["real"] != errores["prediccion"]
    resumen_error = (
        errores.groupby("real")
        .agg(total=("error", "size"), errores=("error", "sum"))
        .assign(tasa_error=lambda x: (x["errores"] / x["total"]).round(4))
    )
    st.markdown("**Error por clase**")
    st.dataframe(resumen_error, width="stretch")
    pred_csv = pd.DataFrame(
        {
            "y_true": resultado.y_true,
            "y_pred": resultado.y_pred,
        }
    )
    st.download_button(
        "Exportar predicciones",
        data=pred_csv.to_csv(index=True).encode("utf-8"),
        file_name=f"predicciones_{resultado.algoritmo}.csv",
        mime="text/csv",
    )


def _render_comparacion(datos: pd.DataFrame, configuracion: dict) -> None:
    """Compara RF y Naive Bayes con la configuración activa."""
    _resumen_dataset_modulo(datos, configuracion["target"])
    st.markdown("### Comparación de modelos")
    if not configuracion["features"]:
        st.warning("Seleccione al menos una feature para comparar.")
        return
    metrica = st.selectbox(
        "Métrica",
        options=[
            "accuracy",
            "precision_global",
            "precision_macro",
            "recall_global",
            "recall_macro",
            "f1_global",
            "f1_macro",
        ],
        index=0,
        key="clasif_metrica_comparar",
    )
    firma = _firma_clasificacion(configuracion, "comparacion", metrica)
    if st.button("Comparar modelos", type="primary"):
        try:
            resultados = []
            for nombre in ("RF", "NR"):
                clasificador = Clasificacion(
                    dataframe=datos,
                    target=configuracion["target"],
                    features=configuracion["features"],
                )
                if nombre == "RF":
                    resultado = clasificador.RF(
                        test_size=configuracion["test_size"],
                        random_state=configuracion["random_state"],
                        stratify=configuracion["estratificar"],
                        incluir_categoricas=configuracion["incluir_categoricas"],
                        imputar=configuracion["imputar"],
                        estandarizar=configuracion["estandarizar"],
                    )
                else:
                    resultado = clasificador.NR(
                        test_size=configuracion["test_size"],
                        random_state=configuracion["random_state"],
                        stratify=configuracion["estratificar"],
                        incluir_categoricas=configuracion["incluir_categoricas"],
                        imputar=configuracion["imputar"],
                        estandarizar=configuracion["estandarizar"],
                    )
                resultados.append(resultado)
            tabla = pd.DataFrame(
                {
                    "algoritmo": [r.algoritmo for r in resultados],
                    metrica: [_valor_metrica(r, metrica) for r in resultados],
                }
            )
            _guardar_resultado("resultado_comparacion_clasificacion", firma, tabla)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible comparar: {exc}")

    tabla = _recuperar_resultado("resultado_comparacion_clasificacion", firma)
    if tabla is None:
        return
    st.dataframe(tabla, width="stretch")
    figura = plt.figure(figsize=(8, 4))
    ejes = figura.add_subplot(1, 1, 1)
    ejes.bar(tabla["algoritmo"], tabla[metrica], color=PALETA["acento"])
    ejes.set_ylabel(metrica)
    ejes.set_title("Comparación")
    _mostrar_figura(figura)


def _render_acp(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta y muestra el análisis de componentes principales."""
    st.markdown("### Análisis de componentes principales")
    _resumen_dataset_modulo(datos)
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


def _render_particional(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta K-Means o K-Medoids y permite evaluar el valor de k."""
    st.markdown("### Agrupamiento particional")
    _resumen_dataset_modulo(datos)
    if len(datos) < 3 or not configuracion["features"]:
        st.warning("Se requieren variables y al menos tres filas.")
        return
    algoritmo = st.selectbox("Algoritmo", ["K-Means", "K-Medoids"])
    limite = min(12, len(datos) - 1)
    clusters = _seleccionar_entero(
        "Número de clusters", 2, limite, 3, key="particional_clusters"
    )
    firma = _firma(configuracion, algoritmo, clusters)
    if st.button("Ejecutar agrupamiento", type="primary"):
        try:
            muestra = _muestrear(datos, configuracion["filas"])
            modelo = _crear_cluster(muestra, configuracion)
            if algoritmo == "K-Means":
                resultado = modelo.K_means(clusters)
            else:
                resultado = modelo.K_medoids(clusters)
            _guardar_resultado("resultado_particional", firma, resultado)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible agrupar el dataset: {exc}")

    resultado = _recuperar_resultado("resultado_particional", firma)
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


def _render_hac(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta clustering jerárquico y presenta el dendrograma."""
    st.markdown("### Clustering jerárquico aglomerativo")
    _resumen_dataset_modulo(datos)
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


def _render_tsne(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta la proyección t-SNE bajo demanda."""
    st.markdown("### Proyección t-SNE")
    _resumen_dataset_modulo(datos)
    filas = min(configuracion["filas"], 2000)
    if filas < 3 or not configuracion["features"]:
        st.warning("t-SNE requiere variables y al menos tres filas.")
        return
    max_perplexity = min(50, filas - 1)
    perplexity = _seleccionar_entero(
        "Perplexity", 2, max_perplexity, 30, key="tsne_perplexity"
    )
    iteraciones = st.slider("Iteraciones", 250, 2000, 1000, 250)
    firma = _firma(configuracion, perplexity, iteraciones, filas)
    if st.button("Ejecutar t-SNE", type="primary"):
        try:
            muestra = _muestrear(datos, filas)
            resultado = _crear_reductor(muestra, configuracion).TSNE(
                perplexity=perplexity, max_iter=iteraciones
            )
            _guardar_resultado("resultado_tsne", firma, resultado)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible ejecutar t-SNE: {exc}")
    resultado = _recuperar_resultado("resultado_tsne", firma)
    if resultado is not None:
        _mostrar_figura(VisualizadorNoSupervisado.proyeccion(resultado))
        st.dataframe(resultado.coordenadas, width="stretch")


def _render_umap(datos: pd.DataFrame, configuracion: dict) -> None:
    """Ejecuta UMAP cuando la dependencia opcional está disponible."""
    st.markdown("### Proyección UMAP")
    _resumen_dataset_modulo(datos)
    filas = min(configuracion["filas"], 3000)
    if filas < 3 or not configuracion["features"]:
        st.warning("UMAP requiere variables y al menos tres filas.")
        return
    max_vecinos = min(100, filas - 1)
    vecinos = _seleccionar_entero(
        "Vecinos", 2, max_vecinos, 15, key="umap_vecinos"
    )
    distancia = st.slider("Distancia mínima", 0.0, 0.99, 0.1, 0.05)
    firma = _firma(configuracion, vecinos, distancia, filas)
    if st.button("Ejecutar UMAP", type="primary"):
        try:
            muestra = _muestrear(datos, filas)
            resultado = _crear_reductor(muestra, configuracion).UMAP(
                n_neighbors=vecinos, min_dist=distancia
            )
            _guardar_resultado("resultado_umap", firma, resultado)
        except DependenciaOpcionalError as exc:
            st.warning(str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"No fue posible ejecutar UMAP: {exc}")
    resultado = _recuperar_resultado("resultado_umap", firma)
    if resultado is not None:
        _mostrar_figura(VisualizadorNoSupervisado.proyeccion(resultado))
        st.dataframe(resultado.coordenadas, width="stretch")


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Framework de análisis de datos",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Framework de análisis y modelos supervisados")
    st.caption(
        "Cambie el CSV y configure las variables; la misma arquitectura ejecuta "
        "EDA, ACP, clasificación supervisada, agrupamiento y proyecciones."
    )

    datos_cargados, etiqueta, identidad = _seleccionar_fuente()
    if datos_cargados is None:
        st.info("Seleccione un CSV local o suba un archivo para comenzar.")
        return
    _sincronizar_dataset(datos_cargados, identidad)
    datos = st.session_state["dataset_preparado"]
    st.markdown(f"**Dataset activo:** `{etiqueta}`")

    configuracion = _configurar_modelos(datos)
    configuracion_clasif = _configurar_modelos_clasificacion(datos)
    dataset, eda, acp, clasificacion, comparacion, particional, hac, tsne, umap = st.tabs(
        [
            "Dataset",
            "EDA",
            "ACP",
            "Clasificación",
            "Comparación de modelos",
            "K-Means",
            "HAC",
            "t-SNE",
            "UMAP",
        ]
    )
    with dataset:
        _render_dataset()
    with eda:
        _render_eda(datos)
    with acp:
        _render_acp(datos, configuracion)
    with clasificacion:
        _render_clasificacion(datos, configuracion_clasif)
    with comparacion:
        _render_comparacion(datos, configuracion_clasif)
    with particional:
        _render_particional(datos, configuracion)
    with hac:
        _render_hac(datos, configuracion)
    with tsne:
        _render_tsne(datos, configuracion)
    with umap:
        _render_umap(datos, configuracion)


if __name__ == "__main__":
    main()
