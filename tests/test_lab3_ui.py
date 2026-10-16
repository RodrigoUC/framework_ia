"""Pruebas headless con datos sintéticos; no sustituyen revisión en navegador."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from streamlit.testing.v1 import AppTest

from framework_ia.datos.particion import ConfiguracionParticion, Particionador
from framework_ia.exportacion_lab03 import construir_archivos_modelo
from framework_ia.modelos.supervisado import Regresion
from framework_ia.ui import streamlit_app
from framework_ia.ui.configuracion_regresion import configuracion_regresion_actual
from framework_ia.ui.estado_regresion import (
    limpiar_resultados_regresion,
    sincronizar_contexto_regresion,
)
from framework_ia.ui.streamlit_app import APP_NAME, VISTAS_REGRESORES
from framework_ia.visualizacion import VisualizadorRegresion

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


def nueva_app():
    x, y = make_regression(n_samples=80, n_features=3, noise=4, random_state=7)
    datos = pd.DataFrame(x, columns=["a", "b", "c"])
    datos["target"] = y
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (datos, "regresion_sintetica.csv", "reg-ui")
    app.session_state["fuente_metadata"] = {
        "sha256": hashlib.sha256(datos.to_csv(index=False).encode()).hexdigest(),
        "fuente": "fixture sintético",
        "identidad": "reg-ui",
    }
    app.run()
    assert not app.exception
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value("regresion").run()
    app.sidebar.button(key="nav_datos").click().run()
    return app


def navegar(app, ruta):
    if ruta in {"regresion_configuracion", "clasificacion_configuracion"}:
        ruta = "configuracion"
    app.sidebar.button(key=f"nav_{ruta}").click().run()
    assert not app.exception


@pytest.mark.parametrize("ruta,algoritmo", VISTAS_REGRESORES.items())
def test_nine_models_train_explicitly_and_export_saved_numeric_evidence(
    ruta, algoritmo
):
    app = nueva_app()
    with patch.object(Regresion, "entrenar", side_effect=AssertionError("auto-fit")):
        navegar(app, ruta)
    assert not any(item.value.startswith("### ") for item in app.markdown)
    app.button(key=f"lab3_entrenar_{algoritmo}").click().run()
    assert not app.exception
    snapshot = app.session_state[f"resultado_lab3_modelo_{algoritmo}"]
    resultado = snapshot["resultado"]
    assert resultado.algoritmo == algoritmo
    assert set(resultado.metricas) == {"rmse", "mae", "mse", "r2"}
    assert all("%" not in metrica.label for metrica in app.metric)
    assert resultado.features == (("a",) if algoritmo == "RLS" else ("a", "b", "c"))
    archivos = construir_archivos_modelo(
        resultado, argumentos=snapshot["argumentos"], dataset=snapshot["dataset"]
    )
    configuracion = json.loads(archivos["configuracion.json"])
    assert (
        configuracion["dataset"]["sha256"]
        == app.session_state["fuente_metadata"]["sha256"]
    )
    with (
        patch.object(Regresion, "entrenar", side_effect=AssertionError("auto-fit")),
        patch(
            "framework_ia.ui.lab3.mostrar_resultado_modelo",
            side_effect=AssertionError("individual diagnostic in comparison"),
        ),
        patch.object(
            Regresion, "experimentar", side_effect=AssertionError("auto-benchmark")
        ),
    ):
        app.session_state["vista_activa"] = "lab3_comparacion"
        app.run()
        assert app.session_state["vista_activa"] == "lab3_resultados"
        navegar(app, "lab3_resultados")
    assert not app.exception
    assert "nav_lab3_comparacion" not in {button.key for button in app.sidebar.button}
    assert not app.metric
    assert not any(widget.label == "Modelo guardado" for widget in app.selectbox)
    assert not any(button.key.startswith("lab3_entrenar_") for button in app.button)
    assert app.session_state[f"resultado_lab3_modelo_{algoritmo}"] is snapshot


def test_parameter_edits_and_failure_keep_success_and_do_not_touch_classification():
    app = nueva_app()
    navegar(app, "regresion_ridge")
    app.button(key="lab3_entrenar_Ridge").click().run()
    previo = app.session_state["resultado_lab3_modelo_Ridge"]
    app.session_state["lab2_param_Ridge_alpha"] = 555
    app.radio(key="lab3_modo_Ridge").set_value("Personalizada").run()
    app.number_input(key="lab3_param_Ridge_alpha").set_value(10.0).run()
    assert app.session_state["resultado_lab3_modelo_Ridge"] is previo
    assert any("sin ejecutar" in aviso.value for aviso in app.info)
    with patch.object(Regresion, "entrenar", side_effect=ValueError("fallo sintético")):
        app.button(key="lab3_entrenar_Ridge").click().run()
    assert not app.exception
    assert app.session_state["resultado_lab3_modelo_Ridge"] is previo
    assert app.session_state["lab2_param_Ridge_alpha"] == 555
    assert any("fallo sintético" in error.value for error in app.error)


def test_configuration_and_read_only_context_change_invalidates_results():
    app = nueva_app()
    navegar(app, "regresion_configuracion")
    assert app.selectbox(key="sup_feature_simple").value == "a"
    assert app.slider(key="sup_test_size").value == 25
    navegar(app, "regresion_rlm")
    app.button(key="lab3_entrenar_RLM").click().run()
    app.session_state["sup_random_state"] = 43
    navegar(app, "lab3_resultados")
    assert "resultado_lab3_modelo_RLM" not in app.session_state
    assert any("Todavía no" in aviso.value for aviso in app.info)


def test_benchmark_partial_winner_failure_and_total_failure_keep_previous_success():
    app = nueva_app()
    navegar(app, "lab3_benchmark")
    assert not any(item.value == "### Experimentos Lab03" for item in app.markdown)
    app.multiselect(key="lab3_algoritmos").set_value(["RLM", "Ridge"]).run()
    app.button(key="lab3_ejecutar_experimento").click().run()
    assert not app.exception
    previo = app.session_state["resultado_lab3_experimento"]
    resultado = previo["resultado"]
    assert len(resultado.tabla) == 3
    ganador = resultado.mejor_algoritmo
    parcial = replace(
        resultado,
        mejores={
            alg: valor for alg, valor in resultado.mejores.items() if alg != ganador
        },
        metadatos={
            **resultado.metadatos,
            "errores_prueba": {ganador: "fallo final sintético"},
        },
    )
    with patch.object(Regresion, "experimentar", return_value=parcial):
        app.button(key="lab3_ejecutar_experimento").click().run()
    assert not app.exception
    assert app.session_state["resultado_lab3_experimento"] is previo
    assert (
        app.session_state["resultado_lab3_intento_experimento"]["resultado"] is parcial
    )
    assert any("falta la evaluación" in aviso.value for aviso in app.warning)
    with patch.object(
        Regresion, "experimentar", side_effect=ValueError("ningún candidato")
    ):
        app.button(key="lab3_ejecutar_experimento").click().run()
    assert not app.exception
    assert app.session_state["resultado_lab3_experimento"] is previo
    assert any("ningún candidato" in error.value for error in app.error)
    with patch.object(
        Regresion, "experimentar", side_effect=AssertionError("auto-fit")
    ):
        navegar(app, "lab3_resultados")


def test_shared_partition_preprocessing_guard_and_source_reset():
    app = nueva_app()
    datos = app.session_state["dataset_preparado"]
    particion = Particionador(
        ConfiguracionParticion(
            porcentaje_test=0.25, porcentaje_validacion=0.2, semilla=13
        )
    ).dividir(datos)
    app.session_state["particion_global"] = particion
    navegar(app, "regresion_configuracion")
    app.checkbox(key="sup_usar_particion_global").check().run()
    navegar(app, "regresion_rlm")
    app.button(key="lab3_entrenar_RLM").click().run()
    assert not app.exception
    resultado = app.session_state["resultado_lab3_modelo_RLM"]["resultado"]
    assert resultado.y_true.index.tolist() == particion.test.index.tolist()
    app.session_state["dataset_preparacion"] = {"imputar": True}
    app.run()
    assert app.button(key="lab3_entrenar_RLM").disabled
    assert any("preparación global" in error.value for error in app.error)
    app.session_state["error_lab3_experimento"] = "anterior"
    app.session_state["fuente_aplicada"] = (datos.copy(), "otro.csv", "otro")
    app.run()
    assert not app.exception
    assert "resultado_lab3_modelo_RLM" not in app.session_state
    assert "error_lab3_experimento" not in app.session_state
    assert not app.session_state["sup_usar_particion_global"]


def test_context_fingerprint_covers_simple_feature_values_and_split_indices():
    datos = pd.DataFrame({"a": range(20), "b": range(20, 40), "target": range(40, 60)})
    estado = {}
    config = configuracion_regresion_actual(datos, estado)
    original = sincronizar_contexto_regresion(estado, datos, config)
    estado["resultado_lab3_modelo_RLM"] = "evidencia"
    assert sincronizar_contexto_regresion(estado, datos, config) == original
    assert estado["resultado_lab3_modelo_RLM"] == "evidencia"
    config["feature_simple"] = "b"
    assert sincronizar_contexto_regresion(estado, datos, config) != original
    assert "resultado_lab3_modelo_RLM" not in estado
    estado.update(
        error_lab3_modelo_RLM="fallo",
        contexto_lab3_modelo_RLM={},
        resultado_lab3_intento_experimento={},
    )
    limpiar_resultados_regresion(estado)
    assert not estado
    assert APP_NAME == "Nexo IA"


def test_default_benchmark_executes_fourteen_candidates_and_saves_selection():
    app = nueva_app()
    navegar(app, "lab3_benchmark")
    app.button(key="lab3_ejecutar_experimento").click().run()
    assert not app.exception
    resultado = app.session_state["resultado_lab3_experimento"]["resultado"]
    assert len(resultado.tabla) == 14
    assert len(resultado.mejores) == 9
    assert resultado.mejor_algoritmo in resultado.mejores
    assert resultado.metadatos["metrica_seleccion"] == "rmse"
    with (
        patch.object(Regresion, "experimentar", side_effect=AssertionError("auto-fit")),
        patch(
            "framework_ia.ui.experimentos_lab3.mostrar_resultado_modelo",
            side_effect=AssertionError("individual diagnostic in results"),
        ),
    ):
        navegar(app, "lab3_resultados")
    assert not app.exception
    assert any("puntuacion_validacion" in frame.value for frame in app.dataframe)
    assert any("rmse" in frame.value for frame in app.dataframe)
    assert not any("real" in chart.key for chart in app.get("plotly_chart"))
    assert len(app.get("download_button")) == 3


def test_individual_regression_page_keeps_only_its_own_diagnostic():
    app = nueva_app()
    for ruta, algoritmo in (("regresion_rls", "RLS"), ("regresion_ridge", "Ridge")):
        navegar(app, ruta)
        app.button(key=f"lab3_entrenar_{algoritmo}").click().run()
    with patch.object(Regresion, "entrenar", side_effect=AssertionError("auto-fit")):
        navegar(app, "regresion_rls")
    assert not app.exception
    assert len(app.metric) == 4
    keys = {chart.key for chart in app.get("plotly_chart")}
    assert keys == {"lab3_RLS_real", "lab3_RLS_residuos"}
    assert len(app.get("download_button")) == 2


def test_invalid_alpha_entry_blocks_training_without_discarding_success():
    app = nueva_app()
    navegar(app, "regresion_lassocv")
    app.button(key="lab3_entrenar_LassoCV").click().run()
    previo = app.session_state["resultado_lab3_modelo_LassoCV"]
    app.radio(key="lab3_modo_LassoCV").set_value("Personalizada").run()
    app.text_input(key="lab3_param_LassoCV_alphas").set_value("0.1, nan").run()
    assert not app.exception
    assert app.button(key="lab3_entrenar_LassoCV").disabled
    assert app.session_state["resultado_lab3_modelo_LassoCV"] is previo


def test_applied_data_split_and_global_clear_remove_all_regression_evidence():
    app = nueva_app()
    navegar(app, "regresion_rlm")
    app.button(key="lab3_entrenar_RLM").click().run()
    app.session_state["error_lab3_experimento"] = "anterior"
    navegar(app, "configuracion")
    app.button(key="particion_calcular").click().run()
    assert not app.exception
    assert "particion_global" in app.session_state
    assert "resultado_lab3_modelo_RLM" not in app.session_state
    assert "error_lab3_experimento" not in app.session_state
    estado = {
        "resultado_lab3_modelo_RLM": {},
        "error_lab3_experimento": "x",
        "regresion_contexto": (),
        "contexto_lab3_modelo_RLM": {},
    }
    with patch.object(streamlit_app.st, "session_state", estado):
        streamlit_app._limpiar_resultados()
    assert not estado


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_regression_figures_use_shared_theme_renderer_with_keys_and_alt(tema):
    resultado = SimpleNamespace(
        y_true=pd.Series([1.0, 2.0, 3.0]),
        y_pred=pd.Series([1.1, 1.9, 3.2]),
        residuos=pd.Series([-0.1, 0.1, -0.2]),
    )
    for figura in (
        VisualizadorRegresion.real_predicho(resultado),
        VisualizadorRegresion.residuos(resultado),
    ):
        with (
            patch.object(
                streamlit_app.st,
                "context",
                SimpleNamespace(theme=SimpleNamespace(type=tema)),
            ),
            patch.object(streamlit_app.st, "plotly_chart") as render,
            patch.object(streamlit_app.st, "caption") as caption,
        ):
            streamlit_app._mostrar_figura(
                figura, key="regression_test", alt="Diagnóstico de regresión"
            )
        assert render.call_args.kwargs["key"] == "regression_test"
        assert render.call_args.kwargs["width"] == "stretch"
        caption.assert_called_once_with("Diagnóstico de regresión")
        assert figura.layout.font.color


def test_branding_page_title_no_data_and_header_share_app_name():
    with (
        patch.object(streamlit_app, "_seleccionar_fuente", return_value=(None, "", "")),
        patch.object(streamlit_app.st, "set_page_config") as config,
        patch.object(streamlit_app.st, "markdown") as markdown,
    ):
        streamlit_app.main()
    assert config.call_args.kwargs["page_title"] == "Nexo IA"
    assert any(call.args == ("# Nexo IA",) for call in markdown.call_args_list)
    app = nueva_app()
    assert any("Nexo IA /" in item.value for item in app.markdown)


def test_experiment_summary_precedes_chart_then_evidence_table_and_downloads():
    from framework_ia.ui import experimentos_lab3

    tabla = pd.DataFrame(
        [
            {
                "algoritmo": "RLM",
                "configuracion": "estandar",
                "seleccionado": True,
                "puntuacion_validacion": 2.5,
                "puntuacion_validacion_std": 0.4,
            }
        ]
    )
    resultado = SimpleNamespace(
        tabla=tabla,
        mejor_algoritmo="RLM",
        mejores={},
        metadatos={"estrategia_seleccion": "cv"},
    )
    snapshot = {"resultado": resultado, "argumentos": {}, "dataset": {}}
    eventos = []
    with (
        patch.object(experimentos_lab3, "st") as interfaz,
        patch.object(
            experimentos_lab3,
            "construir_archivos_experimento",
            return_value={
                "comparacion_validacion.csv": b"csv",
                "configuracion.json": b"{}",
            },
        ),
    ):
        interfaz.session_state = {"resultado_lab3_experimento": snapshot}
        interfaz.columns.return_value = [
            SimpleNamespace(metric=lambda *args: eventos.append("metric"))
            for _ in range(3)
        ]
        interfaz.dataframe.side_effect = lambda *args, **kwargs: eventos.append("table")
        interfaz.download_button.side_effect = lambda *args, **kwargs: eventos.append(
            "download"
        )
        experimentos_lab3.mostrar_experimento_guardado(
            lambda *args, **kwargs: eventos.append("chart")
        )
    assert eventos == [
        "metric",
        "metric",
        "metric",
        "chart",
        "table",
        "download",
        "download",
        "download",
    ]
