"""Dependencias reales del flujo guiado; la disponibilidad nunca entrena."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from framework_ia.datos.particion import ConfiguracionParticion, Particionador
from framework_ia.modelos.supervisado import Clasificacion, Regresion
from framework_ia.datos.preprocesamiento import PreprocesadorSupervisado
from framework_ia.ui.configuracion_clasificacion import (
    configuracion_clasificacion_actual,
)
from framework_ia.ui.configuracion_regresion import configuracion_regresion_actual
from framework_ia.ui.flujo import disponibilidad_supervisada, hay_evidencia


@pytest.fixture
def datos():
    x = np.arange(40, dtype=float)
    return pd.DataFrame({"a": x, "b": x[::-1], "target": x * 1.3 + 2})


def config(datos, familia="regresion", estado=None):
    lector = (
        configuracion_regresion_actual
        if familia == "regresion"
        else configuracion_clasificacion_actual
    )
    return lector(datos, estado or {})


def test_readiness_is_pure_does_not_require_optional_steps_and_never_fits(datos):
    antes = datos.copy(deep=True)
    configuracion = config(datos)
    with (
        patch.object(Regresion, "entrenar", side_effect=AssertionError("fit")),
        patch.object(Clasificacion, "entrenar", side_effect=AssertionError("fit")),
        patch.object(
            PreprocesadorSupervisado, "fit", side_effect=AssertionError("fit")
        ),
    ):
        disponible = disponibilidad_supervisada(datos, configuracion, "regresion")
    assert disponible.lista
    assert configuracion["particion_global"] is None
    assert not configuracion["usar_particion_global"]
    pd.testing.assert_frame_equal(antes, datos)


@pytest.mark.parametrize(
    "target",
    [
        [1.0] * 40,
        [np.nan] + list(range(39)),
        [np.inf] + list(range(39)),
        ["x", "y"] * 20,
    ],
)
def test_invalid_regression_targets_are_blocked_with_configuration_direction(
    datos, target
):
    datos["target"] = target
    resultado = disponibilidad_supervisada(datos, config(datos), "regresion")
    assert not resultado.lista
    assert "Configuración de regresión" in resultado.faltantes[0]


def test_continuous_classification_target_is_not_automatically_converted(datos):
    assert not disponibilidad_supervisada(
        datos, config(datos, "clasificacion"), "clasificacion"
    ).lista
    assert disponibilidad_supervisada(datos, config(datos), "regresion").lista
    datos["target"] = [0, 1] * 20
    assert disponibilidad_supervisada(
        datos, config(datos, "clasificacion"), "clasificacion"
    ).lista


def test_predictor_requirements_and_corrections(datos):
    datos["categoria"] = ["a", "b"] * 20
    estado = {"sup_target": "target", "sup_features": ["a", "categoria"]}
    configuracion = config(datos, estado=estado)
    resultado = disponibilidad_supervisada(datos, configuracion, "regresion")
    assert not resultado.lista and "categoria" in resultado.faltantes[0]
    configuracion["incluir_categoricas"] = True
    assert disponibilidad_supervisada(datos, configuracion, "regresion").lista
    datos.loc[0, "a"] = np.nan
    configuracion["imputar"] = False
    assert not disponibilidad_supervisada(datos, configuracion, "regresion").lista
    configuracion["imputar"] = True
    assert disponibilidad_supervisada(datos, configuracion, "regresion").lista
    assert not disponibilidad_supervisada(
        datos, configuracion, "regresion", preparacion={"imputar": True}
    ).lista


def test_single_feature_only_blocks_multiple_regression_not_other_models(datos):
    configuracion = config(datos, estado={"sup_features": ["a"]})
    assert disponibilidad_supervisada(
        datos, configuracion, "regresion", algoritmo="RLS"
    ).lista
    assert disponibilidad_supervisada(
        datos, configuracion, "regresion", algoritmo="Ridge"
    ).lista
    resultado = disponibilidad_supervisada(
        datos, configuracion, "regresion", algoritmo="RLM"
    )
    assert not resultado.lista and "dos variables" in resultado.faltantes[0]


def test_partition_snapshot_integrity_and_regression_fold_requirements(datos):
    particion = Particionador(ConfiguracionParticion(porcentaje_test=0.25)).dividir(
        datos
    )
    configuracion = config(datos)
    configuracion["particion_global"] = particion
    assert disponibilidad_supervisada(datos, configuracion, "regresion").lista
    configuracion["particion_global"] = replace(
        particion, train=particion.train.iloc[:-1]
    )
    assert not disponibilidad_supervisada(datos, configuracion, "regresion").lista
    pocos = datos.iloc[:8]
    assert disponibilidad_supervisada(
        pocos, config(pocos), "regresion", algoritmo="Ridge"
    ).lista
    assert not disponibilidad_supervisada(
        pocos, config(pocos), "regresion", experimento={"estrategia": "cv"}
    ).lista


def test_classification_missing_targets_and_cv_have_distinct_requirements(datos):
    datos["target"] = [0, 1] * 20
    datos.loc[0, "target"] = np.nan
    configuracion = config(datos, "clasificacion")
    assert disponibilidad_supervisada(datos, configuracion, "clasificacion").lista
    pocos = datos.iloc[1:9]
    configuracion = config(pocos, "clasificacion")
    assert disponibilidad_supervisada(pocos, configuracion, "clasificacion").lista
    resultado = disponibilidad_supervisada(
        pocos,
        configuracion,
        "clasificacion",
        experimento={"estrategia": "cv", "folds": 5},
    )
    assert not resultado.lista and "filas por clase" in resultado.faltantes[0]


def test_results_availability_is_per_family_and_partial_attempts_are_consultable():
    estado = {"resultado_lab3_intento_experimento": {"resultado": "parcial"}}
    assert hay_evidencia(estado, "regresion")
    assert not hay_evidencia(estado, "regresion", solo_modelos=True)
    assert not hay_evidencia(estado, "clasificacion")
    estado["resultado_lab2_modelo_RF"] = object()
    assert hay_evidencia(estado, "clasificacion", solo_modelos=True)


def app_con_datos(datos):
    app = AppTest.from_file(
        Path(__file__).with_name("streamlit_app_entry.py"), default_timeout=60
    )
    app.session_state["fuente_aplicada"] = (
        datos.copy(),
        "flujo_sintetico.csv",
        "flujo-fixture",
    )
    app.run()
    assert not app.exception
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value("regresion").run()
    app.sidebar.button(key="nav_datos").click().run()
    return app


def test_navigation_and_next_steps_follow_actual_dependencies_without_fitting(datos):
    with (
        patch.object(Regresion, "entrenar", side_effect=AssertionError("auto-fit")),
        patch.object(Clasificacion, "entrenar", side_effect=AssertionError("auto-fit")),
        patch.object(
            PreprocesadorSupervisado, "fit", side_effect=AssertionError("auto-fit")
        ),
    ):
        app = app_con_datos(datos)
        assert any("Etapa 1" in item.value for item in app.markdown)
        assert app.sidebar.button(key="nav_clasificacion").disabled
        assert not app.sidebar.button(key="nav_regresion_ridge").disabled
        assert not app.sidebar.button(key="nav_eda").disabled
        assert not app.sidebar.button(key="nav_acp").disabled
        assert not app.sidebar.button(key="nav_kmeans").disabled
        assert app.sidebar.button(key="nav_lab3_resultados").disabled
        app.button(key="flujo_configurar").click().run()
        assert not app.exception
        assert any("Etapa 2" in item.value for item in app.markdown)
        app.multiselect(key="sup_features").set_value([]).run()
        assert app.button(key="flujo_continuar").disabled
        assert app.sidebar.button(key="nav_regresion_ridge").disabled
        assert not app.sidebar.button(key="nav_configuracion").disabled
        app.multiselect(key="sup_features").set_value(["a", "b"]).run()
        app.button(key="flujo_continuar").click().run()
        assert not app.exception
        assert any("Etapa 3" in item.value for item in app.markdown)
        assert not app.button(key="lab3_entrenar_Ridge").disabled


def test_success_enables_results_immediately_and_context_change_revokes_them(datos):
    app = app_con_datos(datos)
    app.sidebar.button(key="nav_regresion_ridge").click().run()
    with patch.object(
        Regresion, "entrenar", autospec=True, side_effect=Regresion.entrenar
    ) as entrenar:
        app.button(key="lab3_entrenar_Ridge").click().run()
    assert not app.exception
    assert entrenar.call_count == 1
    assert not app.sidebar.button(key="nav_lab3_resultados").disabled
    app.button(key="flujo_resultados").click().run()
    assert any("Etapa 4" in item.value for item in app.markdown)
    app.session_state["sup_random_state"] = 21
    app.run()
    assert not app.exception
    assert app.sidebar.button(key="nav_lab3_resultados").disabled
    assert "resultado_lab3_modelo_Ridge" not in app.session_state
    assert not app.button(key="flujo_corregir").disabled


def test_direct_route_cannot_bypass_target_requirements_and_correction_is_available(
    datos,
):
    app = app_con_datos(datos)
    app.session_state["sup_tarea"] = "clasificacion"
    app.session_state["vista_activa"] = "clasificacion_knn"
    with patch.object(Clasificacion, "entrenar") as entrenar:
        app.run()
        assert not app.exception
        assert app.button(key="lab2_entrenar_KNN").disabled
        assert any("objetivo continuo" in item.value for item in app.warning)
        assert not app.button(key="flujo_corregir").disabled
        app.button(key="flujo_corregir").click().run()
        assert app.session_state["vista_activa"] == "configuracion"
        entrenar.assert_not_called()


def test_partial_candidates_do_not_block_a_usable_simple_regression(datos):
    datos["categoria"] = ["a", "b"] * 20
    configuracion = config(
        datos, estado={"sup_target": "target", "sup_features": ["a", "categoria"]}
    )
    opciones = {"estrategia": "cv", "algoritmos": ["RLM", "RLS"]}
    assert disponibilidad_supervisada(
        datos, configuracion, "regresion", experimento=opciones
    ).lista
    opciones["algoritmos"] = ["RLM"]
    assert not disponibilidad_supervisada(
        datos, configuracion, "regresion", experimento=opciones
    ).lista
    app = app_con_datos(datos)
    app.session_state["sup_target"] = "target"
    app.session_state["sup_features"] = ["a", "categoria"]
    app.run()
    assert not app.sidebar.button(key="nav_regresion_rls").disabled
    assert app.sidebar.button(key="nav_regresion_rlm").disabled
    assert not app.sidebar.button(key="nav_lab3_benchmark").disabled
