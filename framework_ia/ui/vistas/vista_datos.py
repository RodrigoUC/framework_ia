"""Vistas de selección de fuente, preparación del dataset, partición y perfil de datos."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from framework_ia.datasets.diabetes import PREDICTORAS_DIABETES
from framework_ia.datos.eda import EDA
from framework_ia.datos.fuentes import CargadorCSV, ConfiguracionCSV
from framework_ia.datos.particion import ConfiguracionParticion, Particionador
from framework_ia.resultados import ResultadoParticion
from framework_ia.ui.controladores.controlador_fuentes import (
    _aplicar_configuracion_diabetes,
    _cargar_fuente_csv,
    _cargar_referencia_regresion,
    _guardar_metadatos_fuente,
    _limpiar_resultados,
)
from framework_ia.ui.controladores.controlador_no_supervisado import (
    _columnas_categoricas,
    _columnas_numericas,
)
from framework_ia.ui.modelos.catalogo_vistas import BASE_DIR
from framework_ia.ui.modelos.estado_regresion import (
    limpiar_resultados_regresion,
)
from framework_ia.ui.modelos.estado_sesion import firma_preparacion
from framework_ia.ui.modelos.perfil_diabetes import (
    RUTA_PROCEDENCIA,
    estado_fuente_diabetes,
)
from framework_ia.ui.vistas.tema import _mostrar_figura
from framework_ia.visualizacion import VisualizadorDatos


def _obtener_base_dir() -> Path:
    """Obtiene el directorio raíz del proyecto permitiendo sustitución en pruebas."""
    modulo_app = sys.modules.get("framework_ia.ui.streamlit_app")
    return getattr(modulo_app, "BASE_DIR", BASE_DIR)


def _obtener_cargador_csv():
    """Obtiene la función de carga de CSV permitiendo sustitución en pruebas."""
    modulo_app = sys.modules.get("framework_ia.ui.streamlit_app")
    return getattr(modulo_app, "_cargar_fuente_csv", _cargar_fuente_csv)


def _seleccionar_fuente() -> tuple[pd.DataFrame | None, str, str]:
    """Permite preparar una fuente y aplicarla al confirmar el formulario."""
    directorio_datos = _obtener_base_dir() / "data"
    archivos = sorted(
        directorio_datos.rglob("*.csv"),
        key=lambda ruta: (
            len(ruta.relative_to(directorio_datos).parts),
            str(ruta.relative_to(directorio_datos)).casefold(),
        ),
    )
    cargar_csv = _obtener_cargador_csv()
    error_inicio = None
    if "fuente_aplicada" not in st.session_state and archivos:
        try:
            cargar_csv(archivos[0], ConfiguracionCSV())
        except Exception as exc:
            error_inicio = str(exc)

    with st.sidebar:
        with st.expander("Fuente de datos", expanded=False):
            if error_inicio:
                st.error(
                    "No fue posible cargar el CSV inicial. Elija otra fuente y "
                    f"pulse «Aplicar fuente». Detalle: {error_inicio}"
                )

            origen = st.radio(
                "Origen",
                ["CSV local", "Subir CSV", "Diabetes · referencia Lab03"],
                key="fuente_origen",
            )
            with st.form("form_fuente_datos"):
                seleccion = None
                archivo = None
                if origen == "Diabetes · referencia Lab03":
                    st.caption("Referencia de regresión scikit-learn (442 filas).")
                else:
                    separador = st.text_input(
                        "Separador",
                        value=",",
                        max_chars=3,
                        key="fuente_separador",
                    )
                    decimal = st.text_input(
                        "Separador decimal",
                        value=".",
                        max_chars=1,
                        key="fuente_decimal",
                    )
                    encoding = st.selectbox(
                        "Codificación",
                        ["utf-8", "latin-1", "cp1252"],
                        key="fuente_encoding",
                    )
                    usar_indice = st.checkbox(
                        "Usar primera columna como índice",
                        key="fuente_usar_indice",
                    )
                    referencia = st.text_input(
                        "Procedencia (URL o referencia, opcional)",
                        key="fuente_referencia",
                    )
                    if origen == "CSV local":
                        if archivos:
                            seleccion = st.selectbox(
                                "Archivo",
                                archivos,
                                format_func=lambda ruta: str(
                                    ruta.relative_to(directorio_datos)
                                ),
                                key="fuente_archivo",
                            )
                        else:
                            st.warning(
                                "No hay archivos CSV en data/. Use la opción de carga."
                            )
                    else:
                        archivo = st.file_uploader(
                            "Archivo CSV", type=["csv"], key="fuente_upload"
                        )
                aplicar = st.form_submit_button(
                    "Aplicar fuente", type="primary"
                )

            if aplicar:
                try:
                    if origen == "Diabetes · referencia Lab03":
                        _cargar_referencia_regresion()
                        return st.session_state["fuente_aplicada"]
                    configuracion = ConfiguracionCSV(
                        separador=separador,
                        decimal=decimal,
                        encoding=encoding,
                        usar_primera_columna_como_indice=usar_indice,
                    )
                    if not separador:
                        raise ValueError(
                            "El separador del CSV no puede estar vacío."
                        )
                    if origen == "CSV local" and seleccion is not None:
                        cargar_csv(seleccion, configuracion, fuente=referencia)
                    elif origen == "Subir CSV" and archivo is not None:
                        contenido = archivo.getvalue()
                        datos = CargadorCSV.cargar_bytes(
                            contenido, configuracion
                        )
                        digest = hashlib.sha256(contenido).hexdigest()
                        st.session_state["fuente_aplicada"] = (
                            datos,
                            archivo.name,
                            f"upload:{digest}:{configuracion}",
                        )
                        _guardar_metadatos_fuente(
                            contenido, configuracion, referencia
                        )
                    else:
                        st.warning(
                            "Seleccione un archivo CSV antes de aplicarlo."
                        )
                except Exception as exc:
                    st.error(f"No fue posible aplicar la fuente: {exc}")
    return st.session_state.get("fuente_aplicada", (None, "", ""))


def _render_dataset() -> None:
    """Permite preparar, restaurar y exportar el dataset persistente."""
    original = st.session_state["dataset_original"]
    actual = st.session_state["dataset_preparado"]
    st.markdown("### Preparación del dataset")

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
        persist_state="session",
        help="Se descartan del dataset original antes de aplicar el resto de la preparación.",
    )
    retenidas = [col for col in original if col not in columnas_eliminar]
    seleccion_nulos = st.session_state.get("dataset_columnas_sin_nulos", [])
    if any(col not in retenidas for col in seleccion_nulos):
        st.session_state["dataset_columnas_sin_nulos"] = [
            col for col in seleccion_nulos if col in retenidas
        ]
    columnas_sin_nulos = st.multiselect(
        "Eliminar filas con nulos en estas columnas",
        options=retenidas,
        default=[],
        key="dataset_columnas_sin_nulos",
        persist_state="session",
        help="Se elimina la fila completa si falta un valor en alguna columna seleccionada. El objetivo sólo se filtra si lo selecciona; no se rellena ni se deduce automáticamente.",
    )
    if columnas_sin_nulos:
        eliminadas = int(original[columnas_sin_nulos].isna().any(axis=1).sum())
        st.caption(
            f"El filtro eliminará {eliminadas} de {len(original)} filas antes de quitar duplicados. Se aplica siempre sobre el original."
        )
    eliminar_duplicados = st.checkbox(
        "Eliminar duplicados",
        value=True,
        key="dataset_eliminar_duplicados",
        persist_state="session",
    )
    with st.expander("Transformaciones globales · solo exploración y clustering"):
        imputar_nulos = st.checkbox(
            "Imputar valores nulos",
            value=False,
            key="dataset_imputar",
            persist_state="session",
        )
        escalado = st.selectbox(
            "Escalado permanente",
            options=["Ninguno", "Normalizar", "Estandarizar"],
            key="dataset_escalado",
            persist_state="session",
        )

    accion, restaurar = st.columns(2)
    if accion.button("Aplicar preparación", type="primary"):
        try:
            eda = EDA(dataframe=original)
            preparado = eda.preparar_dataset(
                columnas_eliminar=columnas_eliminar,
                columnas_sin_nulos=columnas_sin_nulos,
                eliminar_duplicados=eliminar_duplicados,
                imputar_nulos=imputar_nulos,
                normalizar=escalado == "Normalizar",
                estandarizar=escalado == "Estandarizar",
            )
            st.session_state["dataset_preparado"] = preparado.copy()
            st.session_state["dataset_preparacion"] = {
                **eda.preparacion,
                "imputar": imputar_nulos,
                "escalado": escalado,
            }
            _limpiar_resultados()
            st.success("La preparación se aplicó correctamente.")
            st.rerun()
        except Exception as exc:
            st.error(f"No fue posible preparar el dataset: {exc}")

    if restaurar.button("Restaurar dataset original"):
        st.session_state["dataset_preparado"] = original.copy()
        st.session_state.pop("dataset_preparacion", None)
        _limpiar_resultados()
        st.success("Se restauró el contenido original.")
        st.rerun()

    if preparacion := st.session_state.get("dataset_preparacion"):
        st.caption(
            f"Preparación aplicada: {preparacion.get('filas_preparadas', len(actual))} filas · "
            f"Nulos: {preparacion.get('filas_eliminadas_nulos', 0)} filas eliminadas · "
            f"Duplicados: {preparacion.get('filas_eliminadas_duplicados', 0)} filas eliminadas."
        )
    st.dataframe(actual, width="stretch")
    st.download_button(
        "Descargar dataset actual",
        data=actual.to_csv(index=False).encode("utf-8"),
        file_name="dataset_preparado.csv",
        mime="text/csv",
    )


def _render_particion(datos: pd.DataFrame, configuracion: dict) -> None:
    """Calcula una partición train/test/validación reutilizable en el framework."""
    st.markdown("#### División en entrenamiento y prueba")
    st.caption(
        "La partición calculada aquí queda disponible para Clasificación y "
        "Regresión, que pueden reutilizarla en vez de dividir los datos de nuevo."
    )
    porcentaje_test = st.session_state["sup_test_size"]
    porcentaje_val = st.slider(
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
    resumen = (
        f"Train: **{porcentaje_train}%** · Test: **{porcentaje_test}%**"
    )
    if porcentaje_val:
        resumen += f" · Validación: **{porcentaje_val}%**"
    st.caption(resumen)

    columna_seleccionada = (
        configuracion["target"]
        if st.session_state["sup_tarea"] == "clasificacion"
        and st.session_state["sup_strat"]
        else None
    )
    semilla = st.session_state["sup_random_state"]

    if st.button(
        "Calcular partición", type="primary", key="particion_calcular"
    ):
        try:
            cfg = ConfiguracionParticion(
                porcentaje_test=porcentaje_test / 100.0,
                porcentaje_validacion=porcentaje_val / 100.0,
                columna_estratificacion=columna_seleccionada,
                semilla=int(semilla),
            )
            st.session_state["particion_global"] = Particionador(cfg).dividir(
                datos
            )
            st.session_state["sup_particion_firma"] = firma_preparacion(
                st.session_state
            )
            st.session_state.pop("sup_aviso_particion", None)
            limpiar_resultados_regresion(st.session_state)

            for clave in (
                "resultado_modelos_clasificacion",
                "resultado_lab2_experimento",
                "resultado_lab2_intento_experimento",
                "error_lab2_experimento",
                "resultado_lab2_modelo",
                "preview_clasificacion",
                "clasificacion_contexto",
            ):
                st.session_state.pop(clave, None)
            for clave in tuple(st.session_state):
                if clave.startswith(
                    ("resultado_lab2_modelo_", "contexto_lab2_modelo_")
                ):
                    st.session_state.pop(clave, None)
            st.success(
                "Partición calculada y disponible para el resto del framework."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"No fue posible calcular la partición: {exc}")

    particion: ResultadoParticion | None = st.session_state.get(
        "particion_global"
    )
    if particion is None:
        return

    if particion.advertencia:
        st.warning(particion.advertencia)

    m1, m2, m3 = st.columns(3)
    m1.metric("Train", len(particion.train))
    m2.metric("Test", len(particion.test))
    m3.metric(
        "Validación",
        len(particion.validacion) if particion.validacion is not None else 0,
    )

    grafico_izq, grafico_der = st.columns(2)
    with grafico_izq:
        _mostrar_figura(VisualizadorDatos.tamanos_particion(particion))
    with grafico_der:
        grafico_distribucion = VisualizadorDatos.distribucion_particion(
            particion
        )
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


def _resumen_dataset_modulo(
    datos: pd.DataFrame, objetivo: str | None = None
) -> None:
    """Muestra el resumen reutilizable del dataset activo desde cada módulo."""
    eda = EDA(dataframe=datos)
    resumen = eda.resumen_calidad()
    numericas = _columnas_numericas(datos)
    categoricas = _columnas_categoricas(datos)
    outliers = 0
    if numericas:
        outliers = len(
            eda.detectar_outliers(numericas)["filas_con_algun_outlier"]
        )
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


def render_perfil_diabetes(datos: pd.DataFrame) -> None:
    """Ofrece el perfil únicamente para el CSV preparado reconocido e intacto."""
    perfil = estado_fuente_diabetes(datos, st.session_state)
    if perfil == "original":
        st.warning(
            "Este es el CSV original de Diabetes. Para los experimentos Lab02, "
            "aplique el archivo preparado data/lab02/diabetes.csv desde Fuente de datos. "
            "El preparado contiene el objetivo diabetes y conserva las variables nativas."
        )
    elif perfil == "preparado":
        st.caption(
            "Diabetes Lab02: objetivo diabetes, 17 predictoras nativas sin id ni glyhb; "
            "codificación de location, gender y frame, imputación y escalado con train."
        )
        st.button(
            "Aplicar configuración Diabetes Lab02",
            key="aplicar_configuracion_diabetes_lab02",
            on_click=_aplicar_configuracion_diabetes,
        )
        if (
            st.session_state.get("perfil_diabetes_aplicado")
            == st.session_state["fuente_aplicada"][2]
            and st.session_state.get("sup_target") == "diabetes"
            and st.session_state.get("sup_features")
            == list(PREDICTORAS_DIABETES)
        ):
            st.caption(
                "Configuración aplicada. Revise las opciones y ejecute los experimentos cuando esté listo."
            )
