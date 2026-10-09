"""Headless checks of LAB02 navigation, real training and stale-state safety.

All datasets in these tests are synthetic; they are not academic results for
Potabilidad or Diabetes.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from streamlit.testing.v1 import AppTest

from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.ui.estado_clasificacion import huella_datos, sincronizar_contexto

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


def nueva_app():
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=7,
    )
    datos = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    datos["clase"] = y
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (
        datos,
        "fixture_sintetico.csv",
        "synthetic-ui-fixture",
    )
    app.run()
    assert not app.exception
    return app


def ejecutar_comparacion(app):
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    app.multiselect(key="lab2_algoritmos").set_value(["KNN", "DT"]).run()
    app.button(key="lab2_ejecutar").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["resultado_lab2_experimento"]["resultado"].mejores


def test_navigation_separates_configuration_and_read_only_results():
    app = nueva_app()
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert any("Todavía no hay una comparación" in alerta.value for alerta in app.info)
    assert not app.dataframe
    ejecutar_comparacion(app)
    with patch.object(
        Clasificacion,
        "experimentar",
        side_effect=AssertionError("Results must not train"),
    ):
        app.button(key="lab2_ver_resultados").click().run()
        assert not app.exception
        assert not any(s.key == "clasif_target" for s in app.selectbox)
        tabla = app.dataframe[0].value
        assert len(tabla) == 6
        assert {"accuracy_validacion", "f1_macro_validacion", "parametros"} <= set(
            tabla
        )
        assert tabla["seleccionado"].sum() == 2
        resumen = app.dataframe[1].value
        modelos = app.session_state["resultado_lab2_experimento"]["resultado"].mejores
        for _, fila in resumen.iterrows():
            assert (
                fila["F1 macro prueba"]
                == modelos[fila["Algoritmo"]].metricas["f1"]["macro"]
            )
        modelo = modelos[app.selectbox(key="lab2_resultado_algoritmo").value]
        f1 = next(m for m in app.metric if m.label == "F1 macro")
        assert float(f1.value) == pytest.approx(
            modelo.metricas["f1"]["macro"], abs=0.0001
        )


def test_clustering_navigation_has_distinct_algorithms_and_embedded_projections():
    app = nueva_app()
    labels = [button.label for button in app.sidebar.button]
    assert "EDA y ACP" in labels
    assert "K-Means" in labels
    assert "K-Medoids" in labels
    assert "HAC" in labels
    assert "K-Means y K-Medoids" not in labels
    assert not any(
        button.key in {"nav_acp", "nav_tsne", "nav_umap"}
        for button in app.sidebar.button
    )

    app.sidebar.button(key="nav_kmedoids").click().run()
    assert not app.exception
    assert any(button.label == "Ejecutar K-Medoids" for button in app.button)
    assert not any(button.label == "Ejecutar t-SNE" for button in app.button)

    app.sidebar.button(key="nav_kmeans").click().run()
    assert not app.exception
    assert any(button.label == "Ejecutar K-Means" for button in app.button)
    assert any(button.label == "Ejecutar t-SNE" for button in app.button)
    assert any(button.label == "Ejecutar UMAP" for button in app.button)
    assert "resultado_kmeans_tsne" not in app.session_state
    assert "resultado_kmeans_umap" not in app.session_state


def test_legacy_acp_destination_migrates_to_contextual_eda():
    app = nueva_app()
    app.session_state["vista_activa"] = "acp"
    app.run()
    assert not app.exception
    assert app.session_state["vista_activa"] == "eda"
    assert any(button.key == "nav_eda" for button in app.sidebar.button)
    assert any(button.label == "Ejecutar ACP" for button in app.button)


def test_benchmark_controls_persist_and_changes_invalidate_results():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.button(key="lab2_ver_resultados").click().run()
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    assert not app.exception
    assert app.multiselect(key="lab2_algoritmos").value == ["KNN", "DT"]
    assert "resultado_lab2_experimento" in app.session_state
    assert any(
        "Entrenamiento: 60% · Validación: 15% · Prueba: 25%" in c.value
        for c in app.caption
    )
    app.slider(key="lab2_validacion").set_value(25).run()
    assert "resultado_lab2_experimento" not in app.session_state
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert any("debe ejecutar de nuevo" in i.value for i in app.info)


@pytest.mark.parametrize(
    "clave,valor", [("clasif_test_size", 30), ("clasif_random_state", 9)]
)
def test_split_and_seed_changes_invalidate_benchmark(clave, valor):
    app = nueva_app()
    ejecutar_comparacion(app)
    app.get_by_key(clave).set_value(valor).run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state


def test_features_and_source_changes_invalidate_benchmark():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.multiselect(key="clasif_features").set_value(["a", "b"]).run()
    assert "resultado_lab2_experimento" not in app.session_state
    app.button(key="lab2_ejecutar").click().run()
    assert "resultado_lab2_experimento" in app.session_state
    datos, nombre, identidad = app.session_state["fuente_aplicada"]
    app.session_state["fuente_aplicada"] = (datos.copy(), nombre, identidad + "-new")
    app.run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state


def test_global_imputation_blocks_lab2_until_source_is_restored():
    app = nueva_app()
    app.session_state["dataset_preparacion"] = {"imputar": True, "escalado": "Ninguno"}
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    assert app.button(key="lab2_ejecutar").disabled
    assert any("fuga de información" in e.value for e in app.error)
    app.sidebar.button(key="nav_lab2_modelo").click().run()
    assert app.button(key="lab2_entrenar").disabled
    app.sidebar.button(key="nav_datos").click().run()
    next(b for b in app.button if b.label == "Restaurar dataset original").click().run()
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    assert not app.button(key="lab2_ejecutar").disabled


def test_invalid_json_and_empty_selection_cannot_train():
    app = nueva_app()
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    app.text_area(key="lab2_config_json").set_value('{"KNN": []}').run()
    assert app.button(key="lab2_ejecutar").disabled
    assert any("al menos una configuración" in e.value for e in app.error)
    app.text_area(key="lab2_config_json").set_value("").run()
    app.multiselect(key="lab2_algoritmos").set_value([]).run()
    assert app.button(key="lab2_ejecutar").disabled
    assert not app.exception


@pytest.mark.parametrize("algoritmo", ["KNN", "DT", "RF", "XGBoost", "AdaBoost"])
def test_individual_models_train_and_show_numeric_confusion_alternative(algoritmo):
    app = nueva_app()
    app.sidebar.button(key="nav_lab2_modelo").click().run()
    app.selectbox(key="lab2_algoritmo").set_value(algoritmo).run()
    app.button(key="lab2_entrenar").click().run()
    assert not app.exception
    assert not app.error
    resultado = app.session_state["resultado_lab2_modelo"]
    assert app.dataframe[0].value.equals(resultado.matriz_confusion)
    app.radio(key=f"lab2_modo_{algoritmo}").set_value("Personalizada").run()
    assert "resultado_lab2_modelo" not in app.session_state


def test_result_signature_detects_content_and_target_changes():
    datos = pd.DataFrame({"a": [1, 2], "clase": [0, 1]})
    configuracion = {
        "target": "clase",
        "features": ["a"],
        "test_size": 0.25,
        "random_state": 42,
        "estratificar": True,
        "incluir_categoricas": False,
        "imputar": True,
        "estandarizar": True,
    }
    estado = {}
    sincronizar_contexto(estado, datos, configuracion)
    estado["resultado_lab2_experimento"] = object()
    modificado = datos.copy()
    modificado.loc[0, "a"] = 9
    assert huella_datos(datos) != huella_datos(modificado)
    sincronizar_contexto(estado, modificado, configuracion)
    assert "resultado_lab2_experimento" not in estado
    estado["resultado_lab2_experimento"] = object()
    sincronizar_contexto(
        estado, modificado, {**configuracion, "target": "a", "features": ["clase"]}
    )
    assert "resultado_lab2_experimento" not in estado


def test_test_evaluation_failures_are_visible_without_expanding_metadata():
    from dataclasses import replace

    app = nueva_app()
    ejecutar_comparacion(app)
    guardado = dict(app.session_state["resultado_lab2_experimento"])
    resultado = guardado["resultado"]
    guardado["resultado"] = replace(
        resultado,
        mejores={},
        metadatos={
            **resultado.metadatos,
            "errores_prueba": {"KNN": "Valor faltante en prueba"},
        },
    )
    app.session_state["resultado_lab2_experimento"] = guardado
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert not app.exception
    assert any("No se pudo evaluar en prueba" in e.value for e in app.error)
    assert any("Valor faltante en prueba" in e.value for e in app.warning)


def test_applying_a_new_global_split_invalidates_read_only_results():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.sidebar.button(key="nav_datos").click().run()
    app.button(key="particion_calcular").click().run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert any("Todavía no hay una comparación" in e.value for e in app.info)
