"""Una preparación común, tarea explícita y entrenamiento independiente."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from ai_framework.models.supervised import Clasificacion, Regresion
from ai_framework.ui.models.classification_state import (
    configuracion_clasificacion_actual,
)
from ai_framework.ui.models.regression_state import configuracion_regresion_actual
from ai_framework.ui.models.session_state import (
    firma_preparacion,
    inicializar_configuracion,
)
from ai_framework.ui.models.availability import disponibilidad_supervisada


def nueva_app():
    app = AppTest.from_file(
        Path(__file__).with_name("streamlit_app_entry.py"), default_timeout=60
    )
    app.session_state["fuente_aplicada"] = (
        pd.DataFrame({"x": range(40), "z": range(40, 80), "y": [0, 1] * 20}),
        "tareas.csv",
        "tareas",
    )
    return app.run()


def test_task_is_explicit_and_common_configuration_is_unique_without_fitting():
    with (
        patch.object(Clasificacion, "entrenar") as clasificar,
        patch.object(Regresion, "entrenar") as regresar,
    ):
        app = nueva_app()
        assert not app.exception
        assert app.session_state["sup_tarea"] is None
        assert app.sidebar.button(key="nav_clasificacion_knn").disabled
        assert app.sidebar.button(key="nav_regresion_rls").disabled
        assert "Seleccione la tarea" in app.sidebar.button(key="nav_clasificacion_knn").proto.help
        assert "Ejecute primero" in app.sidebar.button(key="nav_lab3_resultados").proto.help
        assert not any("Seleccione la tarea" in item.value or "ejecute primero" in item.value.lower() for item in app.sidebar.caption)
        assert len([b for b in app.sidebar.button if b.label == "Configuración"]) == 1
        app.sidebar.button(key="nav_configuracion").click().run()
        app.radio(key="sup_tarea").set_value("clasificacion").run()
        assert not app.sidebar.button(key="nav_clasificacion_knn").disabled
        assert not app.sidebar.button(key="nav_clasificacion_knn").proto.help
        assert app.sidebar.button(key="nav_regresion_rls").disabled
        app.number_input(key="sup_random_state").set_value(17).run()
        app.radio(key="sup_tarea").set_value("regresion").run()
        assert app.number_input(key="sup_random_state").value == 17
        assert not app.sidebar.button(key="nav_regresion_rls").disabled
        assert app.sidebar.button(key="nav_clasificacion_knn").disabled
        assert (
            len([s for s in app.selectbox if s.label == "Variable objetivo (target)"])
            == 1
        )
        assert not any(b.label == "Entrenar modelo" for b in app.button)
        app.session_state["vista_activa"] = "clasificacion_knn"
        app.run()
        assert app.button(key="lab2_entrenar_KNN").disabled
        assert any("Seleccione la tarea" in w.value for w in app.warning)
        clasificar.assert_not_called()
        regresar.assert_not_called()


def test_task_switch_discards_evidence_and_changed_context_discards_external_split():
    app = nueva_app()
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value("regresion").run()
    app.button(key="particion_calcular").click().run()
    assert "particion_global" in app.session_state
    app.checkbox(key="sup_usar_particion_global").check().run()
    assert app.slider(key="sup_test_size").disabled
    app.multiselect(key="sup_features").set_value(["x"]).run()
    assert "particion_global" not in app.session_state
    assert not app.session_state["sup_usar_particion_global"]
    app.sidebar.button(key="nav_regresion_ridge").click().run()
    app.button(key="lab3_entrenar_Ridge").click().run()
    assert "resultado_lab3_modelo_Ridge" in app.session_state
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value("clasificacion").run()
    assert "resultado_lab3_modelo_Ridge" not in app.session_state
    assert not app.exception


def test_legacy_settings_migrate_once_and_both_adapters_read_same_common_state():
    datos = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    estado = {
        "vista_activa": "regresion_configuracion",
        "reg_target": "y",
        "reg_random_state": 17,
    }
    inicializar_configuracion(datos, estado)
    assert estado["sup_tarea"] == "regresion"
    assert estado["sup_random_state"] == 17
    estado["reg_random_state"] = 99
    estado["sup_random_state"] = 23
    inicializar_configuracion(datos, estado)
    a = configuracion_clasificacion_actual(datos, estado)
    b = configuracion_regresion_actual(datos, estado)
    for clave in (
        "target",
        "features",
        "test_size",
        "random_state",
        "imputar",
        "estandarizar",
    ):
        assert a[clave] == b[clave]
    assert b["random_state"] == 23
    assert not disponibilidad_supervisada(datos, a, "clasificacion").lista
    assert disponibilidad_supervisada(datos, b, "regresion").lista


def test_each_shared_preparation_change_rejects_the_previous_external_partition():
    datos = pd.DataFrame({"x": range(20), "z": range(20, 40), "y": [0, 1] * 10})
    for clave, valor in {
        "tarea": "clasificacion",
        "target": "z",
        "features": ["x"],
        "test_size": 30,
        "random_state": 9,
        "incluir_categoricas": True,
        "imputar": False,
        "escalar": False,
        "feature_simple": "z",
    }.items():
        estado = {"sup_tarea": "regresion"}
        inicializar_configuracion(datos, estado)
        estado["particion_global"] = object()
        estado["sup_particion_firma"] = firma_preparacion(estado)
        estado["sup_usar_particion_global"] = True
        estado[f"sup_{clave}"] = valor
        inicializar_configuracion(datos, estado)
        assert "particion_global" not in estado, clave
        assert not estado["sup_usar_particion_global"], clave


def test_legacy_configuration_routes_share_one_destination_without_changing_task():
    app = nueva_app()
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value("regresion").run()
    with patch.object(Regresion, "entrenar") as entrenar:
        for ruta in (
            "regresion",
            "regresion_configuracion",
            "clasificacion_configuracion",
        ):
            app.session_state["vista_activa"] = ruta
            app.run()
            assert app.session_state["vista_activa"] == "configuracion"
            assert app.radio(key="sup_tarea").value == "regresion"
            assert not app.exception
        entrenar.assert_not_called()
