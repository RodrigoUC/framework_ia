"""Individual training and result consultation views for regression."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.utils.supervised.export import (
    build_model_files,
    build_experiment_zip,
    construir_archivos_modelo,
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
    show_missing_data,
    mostrar_faltantes,
)
from ai_framework.ui.views.theme import (
    _get_tabs,
    _obtener_pestanas,
    render_key_value_table,
    render_tabla_clave_valor,
)
from ai_framework.ui.views.controls_view import (
    regression_parameters,
    parametros_regresion,
)
from ai_framework.utils.supervised.regression_config import (
    ALGORITMOS_LAB3,
    NOMBRES_REGRESORES,
)
from ai_framework.utils.visualization import (
    RegressionVisualizer,
    VisualizadorRegresion,
)


def safe_regression_preparation() -> bool:
    """Verify that previous preparation has not imputed or scaled before the split."""
    preparacion = st.session_state.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        st.error(
            "La preparación global imputó o escaló antes del split. Desactive esas transformaciones en Datos y preparación y vuelva a aplicar sus opciones de limpieza. Configure imputación y escalado en Configuración para aprenderlos solo con train."
        )
        return False
    return True


def display_model_result(
    snapshot, mostrar_figura, *, clave: str
) -> None:
    """Render metrics, fit plots, and downloads for regression model."""
    resultado = snapshot["resultado"]
    st.markdown(
        f"#### {NOMBRES_REGRESORES[resultado.algoritmo]}"
    )
    columnas = st.columns(4)
    for columna, nombre in zip(columnas, ("rmse", "mae", "mse", "r2")):
        columna.metric(nombre.upper(), f"{resultado.metricas[nombre]:.4f}")
    st.dataframe(
        pd.DataFrame(
            {"Train": resultado.metricas_train, "Test": resultado.metricas}
        ),
        width="stretch",
    )
    mostrar_figura(
        RegressionVisualizer.real_predicho(resultado),
        key=f"{clave}_real",
        alt="Dispersión de valores reales y predichos en prueba; la diagonal indica predicción perfecta.",
    )
    mostrar_figura(
        RegressionVisualizer.residuos(resultado),
        key=f"{clave}_residuos",
        alt="Residuos frente a predicción en prueba, con referencia horizontal en cero.",
    )
    with st.expander("Contexto ejecutado, parámetros y trazabilidad"):
        tab_params, tab_meta, tab_args, tab_data = _get_tabs(
            [
                "Parámetros del modelo",
                "Metadatos",
                "Argumentos de ejecución",
                "Información del dataset",
            ]
        )
        with tab_params:
            render_key_value_table(
                resultado.parametros,
                columna_clave="Parámetro",
                columna_valor="Valor configurado",
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
    archivos = build_model_files(
        resultado,
        argumentos=snapshot["argumentos"],
        dataset=snapshot["dataset"],
    )
    st.download_button(
        t("sup_descargar_evidencia", "Descargar evidencia del modelo (ZIP)"),
        build_experiment_zip(archivos),
        file_name=f"lab03_{resultado.algoritmo}.zip",
        mime="application/zip",
        key=f"{clave}_zip",
    )
    st.download_button(
        "Predicciones de prueba (CSV)",
        archivos["predicciones_test.csv"],
        file_name=f"{resultado.algoritmo}_test.csv",
        mime="text/csv",
        key=f"{clave}_csv",
    )


def render_regression_model(
    datos, configuracion, mostrar_figura, *, algoritmo: str
) -> None:
    """Render controls and execute training of an individual regressor."""
    segura = safe_regression_preparation()
    parametros = regression_parameters(algoritmo)
    disponibilidad = supervised_availability(
        datos,
        configuracion,
        "regresion",
        algoritmo=algoritmo,
        parametros=parametros,
        preparacion=st.session_state.get("dataset_preparacion"),
    )
    show_missing_data(disponibilidad)
    clave = f"resultado_lab3_modelo_{algoritmo}"
    error = f"error_lab3_modelo_{algoritmo}"
    sin_simple = algoritmo == "RLS" and configuracion["feature_simple"] is None
    if sin_simple:
        st.warning("Seleccione un predictor numérico para RLS en Configuración.")
    if (
        st.button(
            t("sup_entrenar_modelo", "Entrenar modelo"),
            key=f"lab3_entrenar_{algoritmo}",
            type="primary",
            disabled=not segura
            or parametros is None
            or sin_simple
            or not disponibilidad.lista,
        )
        and disponibilidad.lista
    ):
        try:
            with st.spinner("Ajustando regresión solo con entrenamiento…"):
                resultado = create_regressor(
                    datos, configuracion, algoritmo=algoritmo
                ).entrenar(
                    algoritmo=algoritmo,
                    **common_arguments(configuracion),
                    **parametros,
                )
            st.session_state[clave] = execution_snapshot(
                resultado, datos, configuracion, parametros
            )
            st.session_state.pop(error, None)
            st.rerun()
        except (ValueError, TypeError, RuntimeError) as exc:
            st.session_state[error] = str(exc)
    if st.session_state.get(error):
        st.error(
            f"No se completó el intento: {st.session_state[error]}. La ejecución anterior se conserva si existe."
        )
    snapshot = st.session_state.get(clave)
    if snapshot:
        if snapshot["firma_parametros"] != parameters_fingerprint(parametros):
            st.info(
                "Parámetros modificados sin ejecutar. Se muestra la última ejecución exitosa, no estos ajustes pendientes."
            )
        display_model_result(
            snapshot, mostrar_figura, clave=f"lab3_{algoritmo}"
        )
    else:
        st.info(
            t("sup_sin_resultados_contexto", "Sin resultados para este contexto. Configure el modelo y pulse Entrenar modelo.")
        )


def render_regression_results(
    mostrar_figura, *, incluir_experimentos=False
) -> None:
    """Display comparison of trained regressors and experimental evidence."""
    hay_experimento = incluir_experimentos and any(
        st.session_state.get(clave)
        for clave in (
            "resultado_lab3_experimento",
            "resultado_lab3_intento_experimento",
            "error_lab3_experimento",
        )
    )
    snapshots = {
        alg: st.session_state[f"resultado_lab3_modelo_{alg}"]
        for alg in ALGORITMOS_LAB3
        if f"resultado_lab3_modelo_{alg}" in st.session_state
    }
    if not snapshots and not hay_experimento:
        st.info("Todavía no hay modelos entrenados en este contexto.")
        return

    from ai_framework.ui.views.results_view import (
        render_results_layout,
        render_layout_resultados,
        render_individual_models_panel,
        render_panel_modelos_individuales,
    )
    from ai_framework.ui.views.regression_experiments_view import (
        display_saved_experiment,
        mostrar_experimento_guardado,
    )

    def _render_regression_table():
        filas = []
        modelos_params = []
        for alg, snap in snapshots.items():
            res = snap["resultado"]
            nombre = NOMBRES_REGRESORES.get(alg, alg)
            filas.append(
                {
                    "Algoritmo": nombre,
                    "rmse": round(float(res.metricas.get("rmse", 0.0)), 4),
                    "mae": round(float(res.metricas.get("mae", 0.0)), 4),
                    "mse": round(float(res.metricas.get("mse", 0.0)), 4),
                    "r2": round(float(res.metricas.get("r2", 0.0)), 4),
                }
            )
            modelos_params.append((nombre, getattr(res, "parametros", {})))

        render_individual_models_panel(
            pd.DataFrame(filas),
            modelos_params,
            titulo="Modelos entrenados individualmente",
        )

    render_results_layout(
        hay_individuales=bool(snapshots),
        hay_comparativa=hay_experimento,
        render_individuales_fn=_render_regression_table,
        render_comparativa_fn=lambda: display_saved_experiment(mostrar_figura, solo_comparacion=True),
        mensaje_sin_datos="Todavía no hay modelos entrenados en este contexto.",
        clave_contexto="regresion",
    )

mostrar_resultado_modelo = display_model_result

# Backward compatibility aliases
preparacion_regresion_segura = safe_regression_preparation
render_modelo_regresion = render_regression_model
render_resultados_regresion = render_regression_results
