"""Limpieza explícita de filas antes del split, sin aprender transformaciones."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from framework_ia.datos.eda import EDA
from framework_ia.datos.particion import ConfiguracionParticion, Particionador
from framework_ia.modelos.supervisado import Clasificacion, Regresion
from framework_ia.datos.preprocesamiento import PreprocesadorSupervisado
from framework_ia.ui.modelos.estado_clasificacion import (
    configuracion_clasificacion_actual,
)
from framework_ia.ui.modelos.estado_regresion import configuracion_regresion_actual
from framework_ia.ui.modelos.disponibilidad import disponibilidad_supervisada


def datos_sin_limpiar():
    x = np.arange(40, dtype=float)
    datos = pd.DataFrame(
        {"x": x, "z": np.sin(x), "descartar": np.nan, "target": x * 1.3 + 2}
    )
    datos.loc[0, "x"] = np.nan
    datos.loc[1, "target"] = np.nan
    return pd.concat([datos, datos.iloc[[2]]], ignore_index=True)


@pytest.mark.parametrize(
    "columnas,filas,nulos_target",
    [([], 40, 1), (["x"], 39, 1), (["x", "target"], 38, 0)],
)
def test_null_filter_only_uses_selected_columns_and_counts_its_policy(
    columnas, filas, nulos_target
):
    original = datos_sin_limpiar()
    antes = original.copy(deep=True)
    eda = EDA(dataframe=original)
    limpio = eda.preparar_dataset(
        columnas_eliminar=["descartar"],
        columnas_sin_nulos=columnas,
        imputar_nulos=False,
    )
    assert len(limpio) == filas
    assert limpio.target.isna().sum() == nulos_target
    assert limpio.index.tolist() == list(range(filas))
    assert eda.preparacion == {
        "columnas_eliminar": ["descartar"],
        "columnas_sin_nulos": columnas,
        "eliminar_duplicados": True,
        "filas_originales": 41,
        "filas_eliminadas_nulos": 40 - filas,
        "filas_eliminadas_duplicados": 1,
        "filas_preparadas": filas,
    }
    pd.testing.assert_frame_equal(original, antes)


def test_legacy_default_imputation_and_duplicate_behavior_is_preserved():
    datos = pd.DataFrame({"x": [1.0, np.nan, 3.0, 3.0], "y": [2, 4, 6, 6]})
    preparado = EDA(dataframe=datos).preparar_dataset()
    assert preparado.x.tolist() == [1.0, 2.0, 3.0]
    assert len(datos) == 4 and datos.x.isna().sum() == 1


@pytest.mark.parametrize("columna", ["ausente", "descartar"])
def test_null_policy_rejects_missing_or_excluded_columns(columna):
    eda = EDA(dataframe=datos_sin_limpiar())
    with pytest.raises(KeyError, match="filtrar nulos inexistentes"):
        eda.preparar_dataset(
            columnas_eliminar=["descartar"],
            columnas_sin_nulos=[columna],
            imputar_nulos=False,
        )
    assert eda.preparacion is None


def test_empty_filter_result_is_rejected_without_publishing_data_or_policy():
    original = datos_sin_limpiar()
    eda = EDA(dataframe=original)
    with pytest.raises(ValueError, match="todas las filas"):
        eda.preparar_dataset(columnas_sin_nulos=["descartar"], imputar_nulos=False)
    assert eda.preparacion is None
    pd.testing.assert_frame_equal(eda.datos, original)


@pytest.mark.parametrize(
    "aprendida",
    [{"imputar": True}, {"escalado": "Normalizar"}, {"escalado": "Estandarizar"}],
)
def test_row_cleanup_does_not_bypass_learned_global_transformation_guard(aprendida):
    eda = EDA(dataframe=datos_sin_limpiar())
    limpio = eda.preparar_dataset(
        columnas_eliminar=["descartar"],
        columnas_sin_nulos=["x", "target"],
        imputar_nulos=False,
    )
    config = configuracion_regresion_actual(limpio, {})
    politica = {**eda.preparacion, "imputar": False, "escalado": "Ninguno"}
    assert disponibilidad_supervisada(
        limpio, config, "regresion", preparacion=politica
    ).lista
    bloqueada = disponibilidad_supervisada(
        limpio, config, "regresion", preparacion={**politica, **aprendida}
    )
    assert not bloqueada.lista
    assert "vuelva a aplicar la limpieza" in bloqueada.faltantes[0]


@pytest.mark.parametrize("familia", ["clasificacion", "regresion"])
def test_safe_cleanup_is_available_for_both_supervised_families(familia):
    eda = EDA(dataframe=datos_sin_limpiar())
    limpio = eda.preparar_dataset(
        columnas_eliminar=["descartar"],
        columnas_sin_nulos=["x", "target"],
        imputar_nulos=False,
    )
    lector = configuracion_regresion_actual
    if familia == "clasificacion":
        limpio["target"] = np.arange(len(limpio)) % 2
        lector = configuracion_clasificacion_actual
    assert disponibilidad_supervisada(
        limpio, lector(limpio, {}), familia, preparacion=eda.preparacion
    ).lista


def nueva_app():
    original = datos_sin_limpiar()
    app = AppTest.from_file(
        Path(__file__).with_name("streamlit_app_entry.py"), default_timeout=60
    )
    app.session_state["fuente_aplicada"] = (original, "limpieza.csv", "limpieza")
    app.run()
    assert not app.exception
    return app, original


def pulsar(app, etiqueta):
    next(b for b in app.button if b.label == etiqueta).click().run()
    assert not app.exception


def test_cleanup_configuration_explicit_regression_failure_and_restore():
    app, original = nueva_app()
    with (
        patch.object(Regresion, "entrenar", side_effect=AssertionError("auto-fit")),
        patch.object(Clasificacion, "entrenar", side_effect=AssertionError("auto-fit")),
        patch.object(
            PreprocesadorSupervisado, "fit", side_effect=AssertionError("auto-fit")
        ),
    ):
        assert app.multiselect(key="dataset_columnas_sin_nulos").value == []
        app.multiselect(key="dataset_columnas_eliminar").set_value(["descartar"]).run()
        assert (
            "descartar" not in app.multiselect(key="dataset_columnas_sin_nulos").options
        )
        app.multiselect(key="dataset_columnas_sin_nulos").set_value(["x"]).run()
        pulsar(app, "Aplicar preparación")
        assert len(app.session_state["dataset_preparado"]) == 39
        assert app.session_state["dataset_preparado"].target.isna().sum() == 1
        app.sidebar.button(key="nav_configuracion").click().run()
        app.radio(key="sup_tarea").set_value("regresion").run()
        assert app.sidebar.button(key="nav_regresion_ridge").disabled
        assert "nulos" in app.sidebar.button(key="nav_regresion_ridge").proto.help
        app.sidebar.button(key="nav_datos").click().run()
        app.multiselect(key="dataset_columnas_sin_nulos").set_value(
            ["x", "target"]
        ).run()
        pulsar(app, "Aplicar preparación")
        assert len(app.session_state["dataset_preparado"]) == 38
        assert not app.session_state["dataset_preparacion"]["imputar"]
        app.sidebar.button(key="nav_configuracion").click().run()
        assert app.selectbox(key="sup_target").value == "target"
        assert app.multiselect(key="sup_features").value == ["x", "z"]
        assert not app.sidebar.button(key="nav_regresion_ridge").disabled
        app.sidebar.button(key="nav_regresion_ridge").click().run()
    app.button(key="lab3_entrenar_Ridge").click().run()
    assert not app.exception
    snapshot = app.session_state["resultado_lab3_modelo_Ridge"]
    assert len(snapshot["resultado"].y_true) == 10
    assert len(snapshot["resultado"].y_train) == 28
    assert snapshot["dataset"]["preparacion_previa"]["filas_eliminadas_nulos"] == 2
    assert snapshot["dataset"]["preparacion_previa"]["columnas_sin_nulos"] == [
        "x",
        "target",
    ]
    preparado = app.session_state["dataset_preparado"].copy()
    politica = dict(app.session_state["dataset_preparacion"])
    app.sidebar.button(key="nav_datos").click().run()
    app.multiselect(key="dataset_columnas_eliminar").set_value([]).run()
    app.multiselect(key="dataset_columnas_sin_nulos").set_value(["descartar"]).run()
    pulsar(app, "Aplicar preparación")
    assert any("todas las filas" in error.value for error in app.error)
    assert app.session_state["resultado_lab3_modelo_Ridge"] is snapshot
    assert app.session_state["dataset_preparacion"] == politica
    pd.testing.assert_frame_equal(app.session_state["dataset_preparado"], preparado)

    app.multiselect(key="dataset_columnas_eliminar").set_value(["descartar"]).run()
    assert app.multiselect(key="dataset_columnas_sin_nulos").value == []
    app.multiselect(key="dataset_columnas_sin_nulos").set_value(["target"]).run()
    app.session_state["particion_global"] = Particionador(
        ConfiguracionParticion(porcentaje_test=0.25)
    ).dividir(preparado)
    pulsar(app, "Aplicar preparación")
    assert "particion_global" not in app.session_state
    assert "resultado_lab3_modelo_Ridge" not in app.session_state
    assert app.session_state["dataset_preparado"].x.isna().sum() == 1
    assert snapshot["dataset"]["preparacion_previa"] == politica
    pulsar(app, "Restaurar dataset original")
    assert "dataset_preparacion" not in app.session_state
    pd.testing.assert_frame_equal(app.session_state["dataset_original"], original)
    pd.testing.assert_frame_equal(app.session_state["dataset_preparado"], original)


def test_reapply_safe_cleanup_retains_policy_after_global_imputation():
    app, _ = nueva_app()
    app.multiselect(key="dataset_columnas_eliminar").set_value(["descartar"]).run()
    app.multiselect(key="dataset_columnas_sin_nulos").set_value(["target"]).run()
    next(c for c in app.checkbox if c.label == "Imputar valores nulos").check().run()
    pulsar(app, "Aplicar preparación")
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value("regresion").run()
    assert app.sidebar.button(key="nav_regresion_ridge").disabled
    assert (
        "vuelva a aplicar la limpieza"
        in app.sidebar.button(key="nav_regresion_ridge").proto.help
    )
    app.sidebar.button(key="nav_datos").click().run()
    next(c for c in app.checkbox if c.label == "Imputar valores nulos").uncheck().run()
    pulsar(app, "Aplicar preparación")
    assert app.session_state["dataset_preparacion"]["columnas_eliminar"] == [
        "descartar"
    ]
    assert app.session_state["dataset_preparacion"]["columnas_sin_nulos"] == ["target"]
    assert app.session_state["dataset_preparado"].x.isna().sum() == 1
    assert not app.sidebar.button(key="nav_regresion_ridge").disabled
