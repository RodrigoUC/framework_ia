"""Vistas de selección de fuente, preparación del dataset, partición y perfil de datos."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t, obtener_idioma

from ai_framework.datasets.diabetes import PREDICTORAS_DIABETES
from ai_framework.data.eda import EDA
from ai_framework.data.sources import CargadorCSV, ConfiguracionCSV
from ai_framework.data.partition import ConfiguracionParticion, Particionador
from ai_framework.utils.results import ResultadoParticion
from ai_framework.ui.controllers.sources_controller import (
    _aplicar_configuracion_diabetes,
    _cargar_fuente_csv,
    _cargar_referencia_regresion,
    _guardar_metadatos_fuente,
    _limpiar_resultados,
)
from ai_framework.ui.controllers.unsupervised_controller import (
    _columnas_categoricas,
    _columnas_numericas,
)
from ai_framework.ui.models.view_catalog import BASE_DIR
from ai_framework.ui.models.regression_state import (
    limpiar_resultados_regresion,
)
from ai_framework.ui.models.session_state import firma_preparacion
from ai_framework.ui.models.diabetes_profile import (
    RUTA_PROCEDENCIA,
    estado_fuente_diabetes,
)
from ai_framework.ui.views.theme import _mostrar_figura
from ai_framework.utils.visualization import VisualizadorDatos


def _obtener_base_dir() -> Path:
    """Obtiene el directorio raíz del proyecto permitiendo sustitución en pruebas."""
    for mod_name in (
        "ai_framework.ui.streamlit_app",
        "ai_framework.ui.app",
    ):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "BASE_DIR"):
                val = getattr(mod, "BASE_DIR")
                if val != BASE_DIR:
                    return val
    mod = (
        sys.modules.get("ai_framework.ui.streamlit_app")
        or sys.modules.get("ai_framework.ui.app")
    )
    return getattr(mod, "BASE_DIR", BASE_DIR)


def _obtener_cargador_csv():
    """Obtiene la función de carga de CSV permitiendo sustitución en pruebas."""
    for mod_name in (
        "ai_framework.ui.streamlit_app",
        "ai_framework.ui.app",
    ):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "_cargar_fuente_csv"):
                val = getattr(mod, "_cargar_fuente_csv")
                if val is not _cargar_fuente_csv:
                    return val
    return _cargar_fuente_csv


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
        with st.expander(t("fuente_titulo", "Fuente de datos"), expanded=False):
            if error_inicio:
                st.error(
                    "No fue posible cargar el CSV inicial. Elija otra fuente y "
                    f"pulse «Aplicar fuente». Detalle: {error_inicio}"
                )

            opciones_origen = ["CSV local", "Subir CSV", "Diabetes (referencia)"]
            labels_origen = {
                "CSV local": t("fuente_csv_local", "CSV local"),
                "Subir CSV": t("fuente_subir_csv", "Subir CSV"),
                "Diabetes (referencia)": t("fuente_diabetes_ref", "Diabetes (referencia)"),
            }
            origen = st.radio(
                t("fuente_origen_label", "Origen"),
                opciones_origen,
                format_func=lambda opt: labels_origen.get(opt, opt),
                key="fuente_origen",
            )
            with st.form("form_fuente_datos"):
                seleccion = None
                archivo = None
                if origen in ("Diabetes (referencia)", "Diabetes · referencia Lab03"):
                    st.caption("Referencia de regresión scikit-learn (442 filas).")
                else:
                    separador = st.text_input(
                        t("fuente_separador", "Separador"),
                        value=",",
                        max_chars=3,
                        key="fuente_separador",
                    )
                    decimal = st.text_input(
                        t("fuente_separador_decimal", "Separador decimal"),
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
                    t("fuente_aplicar", "Aplicar fuente"), type="primary"
                )

            if aplicar:
                try:
                    if origen in ("Diabetes (referencia)", "Diabetes · referencia Lab03"):
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
    st.markdown(f"### {t('prep_dataset', 'Preparación del dataset')}")

    st.markdown(f"#### {t('prep_tipos_columna', 'Tipos de columna')}")
    tabla_tipos = EDA(dataframe=actual).resumen_columnas()
    tabla_col, grafico_col = st.columns([3, 2])
    with tabla_col:
        lang = obtener_idioma()
        if lang == "en":
            tabla_vista = tabla_tipos.copy()
            tabla_vista["tipo"] = tabla_vista["tipo"].replace({"Numérica": "Numerical", "Categórica": "Categorical"})
            tabla_vista.index.name = "column"
            tabla_vista.columns = ["dtype", "type", "nulls", "null_percentage", "unique_values"]
            st.dataframe(tabla_vista, width="stretch")
        else:
            st.dataframe(tabla_tipos, width="stretch")
    with grafico_col:
        _mostrar_figura(VisualizadorDatos.tipos_columnas(tabla_tipos))

    st.markdown(f"#### {t('prep_limpieza', 'Preparación y limpieza')}")
    columnas_eliminar = st.multiselect(
        t("prep_columnas_eliminar", "Columnas a eliminar"),
        options=original.columns.tolist(),
        default=[],
        key="dataset_columnas_eliminar",
        persist_state="session",
        help=t("prep_columnas_eliminar_help", "Se descartan del dataset original antes de aplicar el resto de la preparación."),
    )
    retenidas = [col for col in original if col not in columnas_eliminar]
    seleccion_nulos = st.session_state.get("dataset_columnas_sin_nulos", [])
    if any(col not in retenidas for col in seleccion_nulos):
        st.session_state["dataset_columnas_sin_nulos"] = [
            col for col in seleccion_nulos if col in retenidas
        ]
    columnas_sin_nulos = st.multiselect(
        t("prep_filas_sin_nulos", "Eliminar filas con nulos en estas columnas"),
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
        t("prep_eliminar_duplicados", "Eliminar duplicados"),
        value=True,
        key="dataset_eliminar_duplicados",
        persist_state="session",
    )
    with st.expander(t("prep_transf_globales", "Transformaciones globales · solo exploración y clustering")):
        imputar_nulos = st.checkbox(
            t("prep_imputar_nulos", "Imputar valores nulos"),
            value=False,
            key="dataset_imputar",
            persist_state="session",
        )
        opciones_escalado = ["Ninguno", "Normalizar", "Estandarizar"]
        escalado_labels = {
            "Ninguno": t("prep_escalado_ninguno", "Ninguno"),
            "Normalizar": t("prep_escalado_normalizar", "Normalizar"),
            "Estandarizar": t("prep_escalado_estandarizar", "Estandarizar"),
        }
        escalado = st.selectbox(
            t("prep_escalado", "Escalado permanente"),
            options=opciones_escalado,
            format_func=lambda opt: escalado_labels.get(opt, opt),
            key="dataset_escalado",
            persist_state="session",
        )

    accion, restaurar = st.columns(2)
    if accion.button(t("prep_aplicar", "Aplicar preparación"), type="primary"):
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

    if restaurar.button(t("prep_restaurar", "Restaurar dataset original")):
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
        t("prep_descargar", "Descargar dataset actual"),
        data=actual.to_csv(index=False).encode("utf-8"),
        file_name="dataset_preparado.csv",
        mime="text/csv",
    )


def _render_particion(datos: pd.DataFrame, configuracion: dict) -> None:
    """Calcula una partición train/test/validación reutilizable en el framework."""
    st.markdown(f"#### {t('prep_division_titulo', 'División en entrenamiento y prueba')}")
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
        t("prep_calcular_particion", "Calcular partición"), type="primary", key="particion_calcular"
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
    m1.metric(t("part_train", "Train"), len(particion.train))
    m2.metric(t("part_test", "Test"), len(particion.test))
    m3.metric(
        t("part_validacion", "Validación"),
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
        st.markdown(f"**{t('part_vista_train', 'Vista previa de train')}**")
        st.dataframe(particion.train.head(5), width="stretch")
    with vista_test:
        st.markdown(f"**{t('part_vista_test', 'Vista previa de test')}**")
        st.dataframe(particion.test.head(5), width="stretch")

    descarga_train, descarga_test, descarga_val = st.columns(3)
    descarga_train.download_button(
        t("part_descargar_train", "Descargar train.csv"),
        particion.train.to_csv(index=False).encode("utf-8"),
        file_name="train.csv",
        mime="text/csv",
    )
    descarga_test.download_button(
        t("part_descargar_test", "Descargar test.csv"),
        particion.test.to_csv(index=False).encode("utf-8"),
        file_name="test.csv",
        mime="text/csv",
    )
    if particion.validacion is not None:
        descarga_val.download_button(
            t("part_descargar_val", "Descargar validacion.csv"),
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
        f"- {t('ds_filas', 'Filas')}: **{resumen['filas']}**",
        f"- {t('ds_columnas', 'Columnas')}: **{resumen['columnas']}**",
        f"- {t('ds_duplicados', 'Duplicados')}: **{resumen['duplicados']}**",
        f"- {t('ds_nulos', 'Nulos')}: **{resumen['nulos']}**",
        f"- {t('ds_numericas', 'Variables numéricas')}: **{len(numericas)}**",
        f"- {t('ds_categoricas', 'Variables categóricas')}: **{len(categoricas)}**",
        f"- {t('ds_outliers', 'Outliers críticos')}: **{outliers}**",
    ]
    if objetivo:
        cardinalidad = datos[objetivo].nunique(dropna=True)
        lineas.append(f"- {t('ds_cardinalidad_target', 'Cardinalidad de target')}: **{cardinalidad}**")
    with st.expander(t("ds_resumen_activo", "Resumen del dataset activo")):
        for linea in lineas:
            st.markdown(linea)


def render_perfil_diabetes(datos: pd.DataFrame) -> None:
    """Ofrece el perfil únicamente para el CSV preparado reconocido e intacto."""
    perfil = estado_fuente_diabetes(datos, st.session_state)
    if perfil == "original":
        st.warning(
            "Este es el CSV original de Diabetes. "
            "Para clasificación binaria, aplique el archivo preparado data/diabetes.csv desde Fuente de datos."
        )
    elif perfil == "preparado":
        st.caption(
            "Diabetes: objetivo diabetes, 17 predictoras nativas sin id ni glyhb; "
            "codificación de variables categóricas, imputación y escalado ajustados con entrenamiento."
        )
        st.button(
            "Aplicar configuración recomendada para Diabetes",
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
