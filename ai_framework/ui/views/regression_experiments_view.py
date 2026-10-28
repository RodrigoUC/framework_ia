"""Vista de ejecución y evidencia experimental de regresión."""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.utils.supervised.export import (
    construir_archivos_experimento,
    construir_zip_experimento,
)
from ai_framework.ui.controllers.regression_controller import (
    argumentos_comunes,
    crear_regresor,
    snapshot_ejecucion,
)
from ai_framework.ui.models.availability import disponibilidad_supervisada
from ai_framework.ui.models.session_state import firma_parametros
from ai_framework.ui.views.navigation import ir_a, mostrar_faltantes
from ai_framework.ui.views.theme import _obtener_pestanas, render_tabla_clave_valor
from ai_framework.ui.views.regression_view import (
    mostrar_resultado_modelo,
    preparacion_regresion_segura,
)
from ai_framework.utils.supervised.regression_config import (
    ALGORITMOS_LAB3,
    configuraciones_experimento_regresion,
)
from ai_framework.utils.visualization import VisualizadorRegresion

def _sanear_tabla_regresion(tabla: pd.DataFrame) -> pd.DataFrame:
    """Prepara la tabla para serialización segura en Arrow sin fallos de tipo."""
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



def mostrar_experimento_guardado(
    mostrar_figura, *, solo_comparacion=False
) -> None:
    """Presenta los resultados de validación, el ganador y los paquetes de descarga."""
    intento = st.session_state.get("resultado_lab3_intento_experimento")
    if intento:
        st.warning(
            "El último intento fue parcial: falta la evaluación final del ganador seleccionado por validación. No se reemplaza la evidencia exitosa anterior ni se elige otro ganador por test."
        )
        errores_intento = intento["resultado"].metadatos.get("errores_prueba", {})
        if errores_intento:
            render_tabla_clave_valor(
                errores_intento,
                columna_clave="Familia / Algoritmo",
                columna_valor="Detalle del error",
            )
        st.dataframe(_sanear_tabla_regresion(intento["resultado"].tabla), hide_index=True, width="stretch")
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
    columnas[2].metric("Candidatos comparados", len(resultado.tabla))
    mostrar_figura(
        VisualizadorRegresion.comparacion_validacion(resultado.tabla),
        key="lab3_validacion",
        alt="RMSE de validación por candidato con desviación entre pliegues cuando aplica; menor es mejor.",
    )
    with st.expander(t("sup_val_todas_configs", "Validación · todas las configuraciones")):
        st.dataframe(_sanear_tabla_regresion(resultado.tabla), hide_index=True, width="stretch")
    errores = resultado.metadatos.get("errores_prueba")
    if errores:
        st.warning(
            "Algunas familias fallaron en la evaluación final; se muestran los errores, sin sustituir la selección por validación."
        )
        render_tabla_clave_valor(
            errores,
            columna_clave="Familia / Algoritmo",
            columna_valor="Detalle del error",
        )
    with st.expander("Selecciones por familia y contexto ejecutado"):
        tab_meta, tab_args, tab_data = _obtener_pestanas(
            [
                "Metadatos del experimento",
                "Argumentos de ejecución",
                "Información del dataset",
            ]
        )
        with tab_meta:
            render_tabla_clave_valor(
                resultado.metadatos,
                columna_clave="Metadato",
                columna_valor="Detalle",
            )
        with tab_args:
            render_tabla_clave_valor(
                snapshot.get("argumentos", {}),
                columna_clave="Argumento",
                columna_valor="Valor",
            )
        with tab_data:
            render_tabla_clave_valor(
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
        mostrar_resultado_modelo(
            {**snapshot, "resultado": ganador},
            mostrar_figura,
            clave="lab3_ganador",
        )
    archivos = construir_archivos_experimento(
        resultado,
        argumentos=snapshot["argumentos"],
        dataset=snapshot["dataset"],
    )
    st.download_button(
        t("sup_descargar_zip_exp", "Descargar evidencia del experimento (ZIP)"),
        construir_zip_experimento(archivos),
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


def render_experimentos_regresion(datos, configuracion, mostrar_figura) -> None:
    """Renderiza los controles y ejecuta el benchmark de regresión."""
    st.caption("14 configuraciones · 9 algoritmos")
    segura = preparacion_regresion_segura()
    algoritmos = st.multiselect(
        "Modelos a evaluar",
        ALGORITMOS_LAB3,
        default=list(ALGORITMOS_LAB3),
        key="lab3_algoritmos",
        persist_state="session",
    )
    estrategia = st.selectbox(
        t("sup_estrategia_val", "Estrategia de validación"),
        ["Automática", "CV", "Holdout"],
        key="lab3_estrategia",
        persist_state="session",
        help="Automática usa holdout si la partición aplicada tiene validación; en otro caso usa CV de 5 pliegues. CV explícita reserva la validación externa sin usarla.",
    )
    validation_size = (
        st.slider(
            "Validación holdout dentro de train (%)",
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
    catalogo = configuraciones_experimento_regresion()
    disponibilidad = disponibilidad_supervisada(
        datos,
        configuracion,
        "regresion",
        experimento=opciones,
        preparacion=st.session_state.get("dataset_preparacion"),
    )
    mostrar_faltantes(disponibilidad)
    with st.expander("Modelos y configuraciones que se evaluarán"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Algoritmo": alg,
                        "Configuración": candidato["nombre"],
                        "Parámetros": str(candidato["parametros"]),
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
            with st.spinner("Evaluando y comparando modelos por validación…"):
                resultado = crear_regresor(datos, configuracion).experimentar(
                    **argumentos_comunes(configuracion), **opciones
                )
            snapshot = snapshot_ejecucion(
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
    if snapshot and snapshot["firma_parametros"] != firma_parametros(opciones):
        st.info(
            "Opciones modificadas sin ejecutar. La evidencia corresponde al último experimento exitoso."
        )
    if snapshot:
        st.success(
            "Evaluación comparativa completada. Puede consultar la comparativa detallada en la sección de Resultados."
        )
        col_btn, _ = st.columns([1, 1])
        with col_btn:
            st.button(
                "Ir a Resultados de regresión",
                key="btn_ir_a_resultados_regresion",
                type="primary",
                on_click=ir_a,
                args=("lab3_resultados",),
            )
        with st.expander("Ver evidencia y resultados de la evaluación en esta vista"):
            mostrar_experimento_guardado(mostrar_figura, solo_comparacion=True)
    else:
        mostrar_experimento_guardado(mostrar_figura, solo_comparacion=True)
