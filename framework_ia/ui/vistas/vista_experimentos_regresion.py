"""Vista de ejecución y evidencia experimental de regresión."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from framework_ia.exportacion import (
    construir_archivos_experimento,
    construir_zip_experimento,
)
from framework_ia.ui.controladores.controlador_regresion import (
    argumentos_comunes,
    crear_regresor,
    snapshot_ejecucion,
)
from framework_ia.ui.modelos.disponibilidad import disponibilidad_supervisada
from framework_ia.ui.modelos.estado_sesion import firma_parametros
from framework_ia.ui.vistas.navegacion import mostrar_faltantes
from framework_ia.ui.vistas.vista_regresion import (
    mostrar_resultado_modelo,
    preparacion_regresion_segura,
)
from framework_ia.utils.configuracion_regresion import (
    ALGORITMOS_LAB3,
    configuraciones_experimento_regresion,
)
from framework_ia.visualizacion import VisualizadorRegresion


def mostrar_experimento_guardado(
    mostrar_figura, *, solo_comparacion=False
) -> None:
    """Presenta los resultados de validación, el ganador y los paquetes de descarga."""
    intento = st.session_state.get("resultado_lab3_intento_experimento")
    if intento:
        st.warning(
            "El último intento fue parcial: falta la evaluación final del ganador seleccionado por validación. No se reemplaza la evidencia exitosa anterior ni se elige otro ganador por test."
        )
        st.json(intento["resultado"].metadatos.get("errores_prueba", {}))
        st.dataframe(intento["resultado"].tabla, hide_index=True, width="stretch")
    error = st.session_state.get("error_lab3_experimento")
    if error:
        st.error(
            f"No se completó el intento: {error}. Se conserva la evidencia anterior del mismo contexto."
        )
    snapshot = st.session_state.get("resultado_lab3_experimento")
    if not snapshot:
        st.info(
            "Sin experimento completo guardado. Ejecute el benchmark para comparar por validación."
        )
        return
    resultado = snapshot["resultado"]
    st.markdown("### Experimento guardado · selección por RMSE de validación")
    st.caption(
        f"Estrategia: {resultado.metadatos['estrategia_seleccion']} · Ganador por validación: {resultado.mejor_algoritmo}"
    )
    seleccion = resultado.tabla.loc[
        (resultado.tabla["algoritmo"] == resultado.mejor_algoritmo)
        & resultado.tabla["seleccionado"]
    ].iloc[0]
    columnas = st.columns(3)
    columnas[0].metric("Ganador por validación", resultado.mejor_algoritmo)
    columnas[1].metric(
        "RMSE de validación", f"{seleccion['puntuacion_validacion']:.4f}"
    )
    columnas[2].metric("Candidatos comparados", len(resultado.tabla))
    mostrar_figura(
        VisualizadorRegresion.comparacion_validacion(resultado.tabla),
        key="lab3_validacion",
        alt="RMSE de validación por candidato con desviación entre pliegues cuando aplica; menor es mejor.",
    )
    st.dataframe(resultado.tabla, hide_index=True, width="stretch")
    errores = resultado.metadatos.get("errores_prueba")
    if errores:
        st.warning(
            "Algunas familias fallaron en la evaluación final; se muestran los errores, sin sustituir la selección por validación."
        )
        st.json(errores)
    with st.expander("Selecciones por familia y contexto ejecutado"):
        st.json(
            {
                "argumentos": snapshot["argumentos"],
                "dataset": snapshot["dataset"],
                "metadatos": resultado.metadatos,
            }
        )
    ganador = resultado.mejores.get(resultado.mejor_algoritmo)
    if solo_comparacion and resultado.mejores:
        st.markdown("#### Prueba reservada · ganadores seleccionados")
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
        "Descargar evidencia del experimento (ZIP)",
        construir_zip_experimento(archivos),
        file_name="lab03_experimento.zip",
        mime="application/zip",
        key="lab3_experimento_zip",
    )
    st.download_button(
        "Comparación de validación (CSV)",
        archivos["comparacion_validacion.csv"],
        file_name="comparacion_validacion.csv",
        mime="text/csv",
        key="lab3_experimento_csv",
    )
    st.download_button(
        "Configuración ejecutada (JSON)",
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
        "Algoritmos del experimento",
        ALGORITMOS_LAB3,
        default=list(ALGORITMOS_LAB3),
        key="lab3_algoritmos",
        persist_state="session",
    )
    estrategia = st.selectbox(
        "Estrategia de validación",
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
    with st.expander("Candidatos que se ejecutarán"):
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
            "Ejecutar experimento",
            key="lab3_ejecutar_experimento",
            type="primary",
            disabled=not segura or not algoritmos or not disponibilidad.lista,
        )
        and disponibilidad.lista
    ):
        try:
            with st.spinner("Comparando candidatos por validación…"):
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
    mostrar_experimento_guardado(mostrar_figura)
