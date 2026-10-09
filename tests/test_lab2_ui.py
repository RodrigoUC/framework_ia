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
    assert any("Todavía no hay resultados" in alerta.value for alerta in app.info)
    assert not any("accuracy_validacion" in frame.value for frame in app.dataframe)
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
    for model in (
        "KNN",
        "Árbol de decisión",
        "Random Forest",
        "XGBoost",
        "AdaBoost",
        "Naive Bayes",
    ):
        assert model in labels
    assert not any("LAB02" in label or "Modelo individual" in label for label in labels)
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


def test_target_balance_reports_numeric_classes_missing_and_rare_without_mutation():
    from framework_ia.datos.eda import EDA

    datos = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "clase": [0, 0, 1, 1, 2, None]})
    original = datos.copy(deep=True)
    conteos, figura = EDA(dataframe=datos).balance_objetivo("clase")
    assert conteos["Cantidad"].to_dict() == {0.0: 2, 1.0: 2, 2.0: 1}
    assert conteos["Proporción"].sum() == pytest.approx(1.0)
    assert conteos["Proporción"].to_dict() == pytest.approx(
        {0.0: 0.4, 1.0: 0.4, 2.0: 0.2}
    )
    assert figura is not None
    assert datos.equals(original)

    spaced = pd.DataFrame({"clase": [1, 1, 1000, None]})
    _, figura_espaciada = EDA(dataframe=spaced).balance_objetivo("clase")
    assert figura_espaciada.layout.xaxis.type == "category"
    assert list(figura_espaciada.data[0].x) == [1, 1000]

    for target in ("Cantidad", "Proporción"):
        collision = pd.DataFrame({target: [target, target, "rare", None]})
        summary, chart = EDA(dataframe=collision).balance_objetivo(target)
        assert summary["Cantidad"].to_dict() == {target: 2, "rare": 1}
        assert summary["Proporción"].to_dict() == pytest.approx(
            {target: 2 / 3, "rare": 1 / 3}
        )
        assert chart.layout.xaxis.type == "category"
        assert collision[target].isna().sum() == 1

    category = pd.Series(
        pd.Categorical(["seen", "seen", None], categories=["seen", "unseen"])
    )
    observed, _ = EDA(dataframe=pd.DataFrame({"target": category})).balance_objetivo(
        "target"
    )
    assert observed["Cantidad"].to_dict() == {"seen": 2}

    app = nueva_app()
    dataset, name, identity = app.session_state["fuente_aplicada"]
    dataset = dataset.copy()
    dataset.loc[dataset.index[0], "clase"] = None
    dataset.loc[dataset.index[1], "clase"] = 2
    app.session_state["fuente_aplicada"] = (dataset, name, identity + ":missing-target")
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert any("Valores faltantes en target: 1" in item.value for item in app.caption)
    assert any("Clases con dos o menos filas" in item.value for item in app.warning)
    assert not app.exception

    vacios = pd.DataFrame({"clase": [None, None]})
    resumen, figura_vacio = EDA(dataframe=vacios).balance_objetivo("clase")
    assert resumen.empty
    assert figura_vacio is None


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
    assert any("ejecute de nuevo" in i.value for i in app.info)


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


def test_results_route_invalidates_stale_classification_without_rendering_controls():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.session_state["clasif_random_state"] = 99
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state
    assert not any(widget.key == "clasif_target" for widget in app.selectbox)
    assert any("ejecute de nuevo" in alerta.value for alerta in app.info)


def test_results_route_clears_evidence_when_dataset_has_no_target_feature_pair():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.session_state["dataset_preparado"] = app.session_state["dataset_preparado"][
        ["a"]
    ]
    app.session_state["vista_activa"] = "lab2_resultados"
    app.run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state
    assert any("al menos una columna objetivo" in info.value for info in app.info)
    assert not any(
        "accuracy_validacion" in frame.value.columns for frame in app.dataframe
    )


def test_global_imputation_blocks_lab2_until_source_is_restored():
    app = nueva_app()
    app.session_state["dataset_preparacion"] = {"imputar": True, "escalado": "Ninguno"}
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    assert app.button(key="lab2_ejecutar").disabled
    assert any("fuga de información" in e.value for e in app.error)
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert app.button(key="lab2_entrenar_KNN").disabled
    app.sidebar.button(key="nav_datos").click().run()
    next(b for b in app.button if b.label == "Restaurar dataset original").click().run()
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    assert not app.button(key="lab2_ejecutar").disabled


def test_invalid_json_and_empty_selection_cannot_train():
    app = nueva_app()
    app.sidebar.button(key="nav_lab2_experimentos").click().run()
    app.selectbox(key="lab2_modo_variantes").set_value("Editar variantes").run()
    app.text_area(key="lab2_config_json").set_value('{"KNN": []}').run()
    assert app.button(key="lab2_ejecutar").disabled
    assert any("al menos una configuración" in e.value for e in app.error)
    app.text_area(key="lab2_config_json").set_value("").run()
    app.multiselect(key="lab2_algoritmos").set_value([]).run()
    assert app.button(key="lab2_ejecutar").disabled
    assert not app.exception


@pytest.mark.parametrize("algoritmo", ["KNN", "DT", "RF", "XGBoost", "AdaBoost", "NR"])
def test_individual_models_train_and_show_numeric_confusion_alternative(algoritmo):
    app = nueva_app()
    rutas = {
        "KNN": "knn",
        "DT": "dt",
        "RF": "",
        "XGBoost": "xgboost",
        "AdaBoost": "adaboost",
        "NR": "nb",
    }
    ruta = f"nav_clasificacion_{rutas[algoritmo]}".rstrip("_")
    app.sidebar.button(key=ruta).click().run()
    assert not any(widget.label == "Algoritmo" for widget in app.selectbox)
    app.button(key=f"lab2_entrenar_{algoritmo}").click().run()
    assert not app.exception
    assert not app.error
    resultado = app.session_state[f"resultado_lab2_modelo_{algoritmo}"]
    assert any(
        frame.value.equals(resultado.matriz_confusion) for frame in app.dataframe
    )
    if algoritmo != "NR":
        app.radio(key=f"lab2_modo_{algoritmo}").set_value("Personalizada").run()
        assert f"resultado_lab2_modelo_{algoritmo}" not in app.session_state


def test_model_settings_and_results_are_scoped_to_each_classifier():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    app.radio(key="lab2_modo_KNN").set_value("Personalizada").run()
    app.number_input(key="lab2_param_KNN_n_neighbors").set_value(7).run()
    app.button(key="lab2_entrenar_KNN").click().run()
    assert "resultado_lab2_modelo_KNN" in app.session_state

    app.sidebar.button(key="nav_clasificacion").click().run()
    app.button(key="lab2_entrenar_RF").click().run()
    assert "resultado_lab2_modelo_RF" in app.session_state
    assert "resultado_lab2_modelo_KNN" in app.session_state

    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert app.radio(key="lab2_modo_KNN").value == "Personalizada"
    assert app.number_input(key="lab2_param_KNN_n_neighbors").value == 7
    assert "resultado_lab2_modelo_KNN" in app.session_state
    assert "resultado_lab2_modelo_RF" in app.session_state
    assert not app.exception


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


def test_all_candidate_failures_remain_visible_without_claiming_a_winner():
    from dataclasses import replace

    app = nueva_app()
    ejecutar_comparacion(app)
    guardado = dict(app.session_state["resultado_lab2_experimento"])
    resultado = guardado["resultado"]
    tabla = resultado.tabla.copy()
    tabla["estado"] = "error"
    tabla["error"] = "Parámetro incompatible"
    guardado["resultado"] = replace(
        resultado,
        tabla=tabla,
        mejores={},
        mejor_algoritmo=None,
    )
    app.session_state["resultado_lab2_experimento"] = guardado
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert not app.exception
    assert any(
        "configuraciones no terminaron" in warning.value for warning in app.warning
    )
    assert not any("Mejor algoritmo" in item.value for item in app.success)
    assert any("error" in frame.value.columns for frame in app.dataframe)


def test_applying_a_new_global_split_invalidates_read_only_results():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.sidebar.button(key="nav_datos").click().run()
    app.button(key="particion_calcular").click().run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert any("Todavía no hay resultados" in e.value for e in app.info)
