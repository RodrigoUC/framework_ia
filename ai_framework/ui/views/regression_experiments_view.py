"""Execution view and experimental evidence for regression."""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.utils.supervised.export import (
    build_experiment_files,
    build_experiment_zip,
    construir_archivos_experimento,
    construir_zip_experimento,
)
from ai_framework.ui.controllers.regression_controller import (
    common_arguments,
    create_regressor,
    execution_snapshot,
    argumentos_comunes,
    crear_regresor,
    snapshot_ejecucion,
)
from ai_framework.ui.models.availability import (
    supervised_availability,
    disponibilidad_supervisada,
)
from ai_framework.ui.models.session_state import (
    parameters_fingerprint,
    firma_parametros,
)
from ai_framework.ui.views.navigation import (
    go_to,
    show_missing_data,
    ir_a,
    mostrar_faltantes,
)
from ai_framework.ui.views.theme import (
    _get_tabs,
    _obtener_pestanas,
    render_key_value_table,
    render_tabla_clave_valor,
)
from ai_framework.ui.views.regression_view import (
    display_model_result,
    mostrar_resultado_modelo,
    safe_regression_preparation,
    preparacion_regresion_segura,
)
from ai_framework.utils.supervised.regression_config import (
    ALGORITMOS_LAB3,
    regression_experiment_configs,
    configuraciones_experimento_regresion,
)
from ai_framework.utils.visualization import (
    RegressionVisualizer,
    VisualizadorRegresion,
)

def _sanitize_regression_table(tabla: pd.DataFrame) -> pd.DataFrame:
    """Prepare table for safe Arrow serialization without type failures."""
    if not isinstance(tabla, pd.DataFrame) or tabla.empty:
        return tabla
    vista = tabla.copy()
    for col in ("parametros", "features", "puntuaciones_pliegues", "metricas_pliegues"):
        if col in vista.columns:
            vista[col] = vista[col].map(
                lambda x: json.dumps(x, ensure_ascii=False, default=str)
                if isinstance(x, (dict, list, tuple, set))
                else (str(x) if x is not None else "")
            )
    return vista



def display_saved_experiment(
    mostrar_figura, *, solo_comparacion=False
) -> None:
    """Present validation results, winner, and download bundles."""
    intento = st.session_state.get("resultado_lab3_intento_experimento")
    if intento:
        st.warning(
            "El último intento fue parcial: falta la evaluación final del ganador seleccionado por validación. No se reemplaza la evidencia exitosa anterior ni se elige otro ganador por test."
        )
        errores_intento = intento["resultado"].metadatos.get("errores_prueba", {})
        if errores_intento:
            render_key_value_table(
                errores_intento,
                columna_clave="Familia / Algoritmo",
                columna_valor="Detalle del error",
            )
        st.dataframe(_sanitize_regression_table(intento["resultado"].tabla), hide_index=True, width="stretch")
    error = st.session_state.get("error_lab3_experimento")
    if error:
        st.error(
            f"No se completó el intento: {error}. Se conserva la evidencia anterior del mismo contexto."
        )
    snapshot = st.session_state.get("resultado_lab3_experimento")
    if not snapshot:
        st.info(
            "Sin evaluaciones guardadas. Ejecute la Evaluación Comparativa de Modelos para comparar por validación."
        )
        return
    resultado = snapshot["resultado"]
    st.markdown("#### Evaluación comparativa · selección por RMSE de validación")
    st.caption(
        f"Estrategia: {resultado.metadatos['estrategia_seleccion']} · Ganador por validación: {resultado.mejor_algoritmo}"
    )
    seleccion = resultado.tabla.loc[
        (resultado.tabla["algoritmo"] == resultado.mejor_algoritmo)
        & resultado.tabla["seleccionado"]
    ].iloc[0]
    columnas = st.columns(3)
    columnas[0].metric(t("sup_ganador_val", "Ganador por validación"), resultado.mejor_algoritmo)
    columnas[1].metric(
        t("sup_rmse_val", "RMSE de validación"), f"{seleccion['puntuacion_validacion']:.4f}"
    )
    columnas[2].metric(t("sup_candidatos_comparados", "Candidatos comparados"), len(resultado.tabla))
    mostrar_figura(
        RegressionVisualizer.comparacion_validacion(resultado.tabla),
        key="lab3_validacion",
        alt="RMSE de validación por candidato con desviación entre pliegues cuando aplica; menor es mejor.",
    )
    with st.expander(t("sup_val_todas_configs", "Validación · todas las configuraciones")):
        st.dataframe(_sanitize_regression_table(resultado.tabla), hide_index=True, width="stretch")
    errores = resultado.metadatos.get("errores_prueba")
    if errores:
        st.warning(
            "Algunas familias fallaron en la evaluación final; se muestran los errores, sin sustituir la selección por validación."
        )
        render_key_value_table(
            errores,
            columna_clave="Familia / Algoritmo",
            columna_valor="Detalle del error",
        )
    with st.expander(t("sup_selecciones_contexto_expander", "Selecciones por familia y contexto ejecutado")):
        tab_meta, tab_args, tab_data = _get_tabs(
            [
                t("tab_meta_exp", "Metadatos del experimento"),
                t("tab_args_exp", "Argumentos de ejecución"),
                t("tab_data_exp", "Información del dataset"),
            ]
        )
        with tab_meta:
            render_key_value_table(
                resultado.metadatos,
                columna_clave="Metadato",
                columna_valor="Detalle",
            )
        with tab_args:
            render_key_value_table(
                snapshot.get("argumentos", {}),
                columna_clave="Argumento",
                columna_valor="Valor",
            )
        with tab_data:
            render_key_value_table(
                snapshot.get("dataset", {}),
                columna_clave="Propiedad",
                columna_valor="Detalle",
            )
    ganador = resultado.mejores.get(resultado.mejor_algoritmo)
    if solo_comparacion and resultado.mejores:
        st.markdown(f"#### {t('sup_test_reservada', 'Prueba reservada · ganadores seleccionados')}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Algoritmo": algoritmo, **modelo.metricas}
                    for algoritmo, modelo in resultado.mejores.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    if ganador is not None and not solo_comparacion:
        display_model_result(
            {**snapshot, "resultado": ganador},
            mostrar_figura,
            clave="lab3_ganador",
        )
    archivos = build_experiment_files(
        resultado,
        argumentos=snapshot["argumentos"],
        dataset=snapshot["dataset"],
    )
    st.download_button(
        t("sup_descargar_zip_exp", "Descargar evidencia del experimento (ZIP)"),
        build_experiment_zip(archivos),
        file_name="lab03_experimento.zip",
        mime="application/zip",
        key="lab3_experimento_zip",
    )
    st.download_button(
        t("sup_comp_val_csv", "Comparación de validación (CSV)"),
        archivos["comparacion_validacion.csv"],
        file_name="comparacion_validacion.csv",
        mime="text/csv",
        key="lab3_experimento_csv",
    )
    st.download_button(
        t("sup_config_ejecutada_json", "Configuración ejecutada (JSON)"),
        archivos["configuracion.json"],
        file_name="configuracion.json",
        mime="application/json",
        key="lab3_experimento_json",
    )


def render_regression_experiments(datos, configuracion, mostrar_figura) -> None:
    """Render controls and execute regression benchmark."""
    st.caption(t("sup_14_configs_9_algs", "14 configuraciones · 9 algoritmos"))
    segura = safe_regression_preparation()
    algoritmos = st.multiselect(
        t("sup_modelos_a_evaluar", "Modelos a evaluar"),
        ALGORITMOS_LAB3,
        default=list(ALGORITMOS_LAB3),
        key="lab3_algoritmos",
        persist_state="session",
    )
    estrategia = st.selectbox(
        t("sup_estrategia_val", "Estrategia de validación"),
        ["Automática", "CV", "Holdout"],
        format_func=lambda x: {
            "Automática": t("estrategia_auto", "Automática"),
            "CV": t("estrategia_cv_corta", "CV"),
            "Holdout": t("estrategia_holdout_corta", "Holdout"),
        }.get(x, x),
        key="lab3_estrategia",
        persist_state="session",
        help=t(
            "sup_estrategia_reg_help",
            "Automática usa holdout si la partición aplicada tiene validación; en otro caso usa CV de 5 pliegues. CV explícita reserva la validación externa sin usarla.",
        ),
    )
    validation_size = (
        st.slider(
            t("sup_val_holdout_train", "Validación holdout dentro de train (%)"),
            10,
            40,
            20,
            5,
            key="lab3_validation_size",
            persist_state="session",
        )
        / 100
    )
    opciones = {
        "algoritmos": algoritmos,
        "estrategia": {"Automática": None, "CV": "cv", "Holdout": "holdout"}[
            estrategia
        ],
        "folds": 5,
        "validation_size": validation_size,
        "feature_simple": configuracion["feature_simple"],
    }
    catalogo = regression_experiment_configs()
    disponibilidad = supervised_availability(
        datos,
        configuracion,
        "regresion",
        experimento=opciones,
        preparacion=st.session_state.get("dataset_preparacion"),
    )
    show_missing_data(disponibilidad)
    with st.expander(t("sup_modelos_configs_eval_expander", "Modelos y configuraciones que se evaluarán")):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        t("col_algoritmo", "Algoritmo"): alg,
                        t("col_configuracion", "Configuración"): candidato["nombre"],
                        t("col_parametros", "Parámetros"): str(candidato["parametros"]),
                    }
                    for alg in algoritmos
                    for candidato in catalogo[alg]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    if (
        st.button(
            t("res_ejecutar_evaluacion", "Ejecutar evaluación comparativa"),
            key="lab3_ejecutar_experimento",
            type="primary",
            disabled=not segura or not algoritmos or not disponibilidad.lista,
        )
        and disponibilidad.lista
    ):
        try:
            with st.spinner(t("sup_evaluando_comparando_spinner", "Evaluando y comparando modelos por validación…")):
                resultado = create_regressor(datos, configuracion).experimentar(
                    **common_arguments(configuracion), **opciones
                )
            snapshot = execution_snapshot(
                resultado, datos, configuracion, opciones
            )
            if resultado.mejor_algoritmo in resultado.mejores:
                st.session_state["resultado_lab3_experimento"] = snapshot
                st.session_state.pop("resultado_lab3_intento_experimento", None)
            else:
                st.session_state["resultado_lab3_intento_experimento"] = (
                    snapshot
                )
            st.session_state.pop("error_lab3_experimento", None)
            st.rerun()
        except (ValueError, TypeError, RuntimeError) as exc:
            st.session_state["error_lab3_experimento"] = str(exc)
            st.session_state.pop("resultado_lab3_intento_experimento", None)
    snapshot = st.session_state.get("resultado_lab3_experimento")
    if snapshot and snapshot["firma_parametros"] != parameters_fingerprint(opciones):
        st.info(
            t("opciones_modificadas_sin_ejecutar", "Opciones modificadas sin ejecutar. La evidencia corresponde al último experimento exitoso.")
        )
    if snapshot:
        st.success(
            t("eval_comp_completada", "Evaluación comparativa completada. Puede consultar la comparativa detallada en la sección de Resultados.")
        )
        col_btn, _ = st.columns([1, 1])
        with col_btn:
            st.button(
                t("btn_ir_a_res_reg", "Ir a Resultados de regresión"),
                key="btn_ir_a_resultados_regresion",
                type="primary",
                on_click=go_to,
                args=("lab3_resultados",),
            )
        with st.expander(t("expander_evidencia_eval", "Ver evidencia y resultados de la evaluación en esta vista")):
            mostrar_experimento_guardado(mostrar_figura, solo_comparacion=True)
    else:
        mostrar_experimento_guardado(mostrar_figura, solo_comparacion=True)

mostrar_experimento_guardado = display_saved_experiment

# Backward compatibility aliases
_sanear_tabla_regresion = _sanitize_regression_table
render_experimentos_regresion = render_regression_experiments
