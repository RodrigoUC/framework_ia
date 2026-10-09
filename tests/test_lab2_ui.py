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
from types import SimpleNamespace

from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.ui.estado_clasificacion import huella_datos, sincronizar_contexto
from framework_ia.ui.lab2 import _metricas_porcentuales

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


def test_saved_metrics_use_accuracy_and_recall_by_actual_class_percent():
    result = SimpleNamespace(
        y_true=pd.Series(["Y", "Y", "Y", "n", "n", "n"]),
        y_pred=pd.Series(["Y", "n", "n", "n", "n", "Y"]),
        labels=("Y", "n"),
        metricas={"accuracy": 0.5},
    )
    metrics = _metricas_porcentuales(result)
    assert metrics["Accuracy general (%)"] == 50.0
    assert metrics["Recall Y (%)"] == pytest.approx(100 / 3)
    assert metrics["Recall N (%)"] == pytest.approx(200 / 3)
    assert metrics["Recall Y (%)"] != 50.0  # This is recall, not precision.


def test_numeric_and_multiclass_labels_are_literal_and_zero_support_is_unavailable():
    numeric = SimpleNamespace(
        y_true=pd.Series([1, 1, 0]),
        y_pred=pd.Series([1, 0, 0]),
        labels=(0, 1, 2),
        metricas={"accuracy": 2 / 3},
    )
    numeric_metrics = _metricas_porcentuales(numeric)
    assert numeric_metrics["Recall actual 1 (%)"] == 50.0
    assert numeric_metrics["Recall actual 0 (%)"] == 100.0
    assert numeric_metrics["Recall actual 2 (%)"] == "Sin casos evaluados"
    assert not any("Recall Y" in name or "Recall N" in name for name in numeric_metrics)

    multiclass = SimpleNamespace(
        y_true=pd.Series(["cat", "dog", "cat"]),
        y_pred=pd.Series(["dog", "dog", "cat"]),
        labels=("cat", "dog", "bird"),
        metricas={"accuracy": 2 / 3},
    )
    class_metrics = _metricas_porcentuales(multiclass)
    assert class_metrics["Recall actual cat (%)"] == 50.0
    assert class_metrics["Recall actual dog (%)"] == 100.0
    assert class_metrics["Recall actual bird (%)"] == "Sin casos evaluados"


def ejecutar_comparacion(app):
    """Train one model; the comparison route must only read its saved result."""
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    app.button(key="lab2_entrenar_KNN").click().run()
    assert not app.exception
    assert "resultado_lab2_modelo_KNN" in app.session_state


def test_navigation_comparison_is_read_only_and_never_experiments():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    app.button(key="lab2_entrenar_KNN").click().run()
    with (
        patch.object(
            Clasificacion, "entrenar", side_effect=AssertionError("comparison trained")
        ),
        patch.object(
            Clasificacion,
            "experimentar",
            side_effect=AssertionError("comparison experimented"),
        ),
    ):
        app.sidebar.button(key="nav_lab2_experimentos").click().run()
        assert not app.exception
        assert any("KNN" in str(frame.value) for frame in app.dataframe)
        assert not any(widget.key == "lab2_ejecutar" for widget in app.button)
        app.sidebar.button(key="nav_lab2_resultados").click().run()
        assert not app.exception
        assert any("KNN" in str(frame.value) for frame in app.dataframe)


def test_random_forest_criteria_are_retained_independently_and_comparison_is_read_only():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion").click().run()
    with patch.object(
        Clasificacion,
        "experimentar",
        side_effect=AssertionError("comparison experimented"),
    ):
        app.button(key="lab2_entrenar_RF").click().run()
        gini = app.session_state["resultado_lab2_modelo_RF_criterios"]["gini"][
            "resultado"
        ]
        app.selectbox(key="lab2_param_RF_criterion").set_value("entropy").run()
        app.button(key="lab2_entrenar_RF").click().run()
        guardados = app.session_state["resultado_lab2_modelo_RF_criterios"]
        entropy = guardados["entropy"]["resultado"]
        assert guardados["gini"]["resultado"] is gini
        assert entropy.parametros["criterion"] == "entropy"
        app.sidebar.button(key="nav_lab2_experimentos").click().run()
        assert not app.exception
        visible = " ".join(str(expander.label) for expander in app.expander)
        assert "gini" in visible and "entropy" in visible
        assert not any(button.key == "lab2_ejecutar" for button in app.button)


def test_rf_parameter_edits_preserve_saved_execution_until_explicit_training():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion").click().run()
    app.button(key="lab2_entrenar_RF").click().run()
    saved = app.session_state["resultado_lab2_modelo_RF_criterios"]["gini"]["resultado"]
    app.radio(key="lab2_modo_RF").set_value("Personalizada").run()
    assert (
        app.session_state["resultado_lab2_modelo_RF_criterios"]["gini"]["resultado"]
        is saved
    )


def test_failed_rf_attempt_keeps_last_successful_criterion():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion").click().run()
    app.button(key="lab2_entrenar_RF").click().run()
    saved = app.session_state["resultado_lab2_modelo_RF_criterios"]["gini"]["resultado"]
    with patch.object(
        Clasificacion, "entrenar", side_effect=ValueError("synthetic failure")
    ):
        app.button(key="lab2_entrenar_RF").click().run()
    assert (
        app.session_state["resultado_lab2_modelo_RF_criterios"]["gini"]["resultado"]
        is saved
    )
    assert any("synthetic failure" in error.value for error in app.error)


def test_clustering_navigation_has_distinct_algorithms_and_embedded_projections():
    app = nueva_app()
    labels = [button.label for button in app.sidebar.button]
    assert "EDA" in labels
    assert "ACP" in labels
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
    assert labels.index("ACP") < labels.index("K-Means")
    assert not any(
        button.key in {"nav_tsne", "nav_umap"} for button in app.sidebar.button
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
    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    assert any("Valores faltantes en target: 1" in item.value for item in app.caption)
    assert any("Clases con dos o menos filas" in item.value for item in app.warning)
    assert not app.exception

    vacios = pd.DataFrame({"clase": [None, None]})
    resumen, figura_vacio = EDA(dataframe=vacios).balance_objetivo("clase")
    assert resumen.empty
    assert figura_vacio is None


def test_eda_and_acp_remain_independent_destinations():
    app = nueva_app()
    app.sidebar.button(key="nav_eda").click().run()
    assert not app.exception
    assert app.session_state["vista_activa"] == "eda"
    assert any(tab.label == "Histogramas" for tab in app.tabs)

    app.sidebar.button(key="nav_acp").click().run()
    assert not app.exception
    assert app.session_state["vista_activa"] == "acp"
    next(
        button for button in app.button if button.label == "Ejecutar ACP"
    ).click().run()
    assert not app.exception
    assert any(tab.label == "Sobreposición" for tab in app.tabs)


def test_results_route_invalidates_stale_classification_without_rendering_controls():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.session_state["clasif_random_state"] = 99
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert not app.exception
    assert "resultado_lab2_modelo_KNN" not in app.session_state
    assert not any(widget.key == "clasif_target" for widget in app.selectbox)
    assert any("Todavía no hay resultados" in alerta.value for alerta in app.info)


def test_results_route_clears_evidence_when_dataset_has_no_target_feature_pair():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.session_state["dataset_preparado"] = app.session_state["dataset_preparado"][
        ["a"]
    ]
    app.session_state["vista_activa"] = "lab2_resultados"
    app.run()
    assert not app.exception
    assert "resultado_lab2_modelo_KNN" not in app.session_state
    assert any("al menos una columna objetivo" in info.value for info in app.info)
    assert not any(
        "accuracy_validacion" in frame.value.columns for frame in app.dataframe
    )


def test_global_imputation_blocks_lab2_until_source_is_restored():
    app = nueva_app()
    app.session_state["dataset_preparacion"] = {"imputar": True, "escalado": "Ninguno"}
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert app.button(key="lab2_entrenar_KNN").disabled
    assert any("fuga de información" in e.value for e in app.error)
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert app.button(key="lab2_entrenar_KNN").disabled
    app.sidebar.button(key="nav_datos").click().run()
    next(b for b in app.button if b.label == "Restaurar dataset original").click().run()
    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert not app.button(key="lab2_entrenar_KNN").disabled


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
        assert f"resultado_lab2_modelo_{algoritmo}" in app.session_state


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


def test_shared_classification_setup_applies_to_models_and_invalidates_them():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    app.selectbox(key="clasif_target").set_value("clase").run()
    app.multiselect(key="clasif_features").set_value(["a", "b"]).run()
    app.slider(key="clasif_test_size").set_value(30).run()
    app.number_input(key="clasif_random_state").set_value(19).run()

    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    assert not any(widget.key == "clasif_target" for widget in app.selectbox)
    assert not any(widget.key == "clasif_features" for widget in app.multiselect)
    assert not any(widget.key == "clasif_test_size" for widget in app.slider)
    assert not any(widget.key == "clasif_random_state" for widget in app.number_input)
    assert not {"Filas", "Columnas", "Nulos", "Duplicados"} & {
        metric.label for metric in app.metric
    }
    app.button(key="lab2_entrenar_KNN").click().run()
    first = app.session_state["resultado_lab2_modelo_KNN"]

    app.sidebar.button(key="nav_clasificacion").click().run()
    assert not {"Filas", "Columnas", "Nulos", "Duplicados"} & {
        metric.label for metric in app.metric
    }
    app.button(key="lab2_entrenar_RF").click().run()
    second = app.session_state["resultado_lab2_modelo_RF"]
    assert first.target == second.target == "clase"
    assert first.features == second.features == ("a", "b")
    assert first.y_true.index.equals(second.y_true.index)

    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    app.number_input(key="clasif_random_state").set_value(20).run()
    assert "resultado_lab2_modelo_KNN" not in app.session_state
    assert "resultado_lab2_modelo_RF" not in app.session_state
    assert "resultado_lab2_modelo_KNN" not in app.session_state
    assert not app.exception


def test_reused_global_partition_is_shared_across_classifier_routes():
    from framework_ia.datos.particion import Particionador
    from framework_ia.utils import ConfiguracionParticion

    app = nueva_app()
    datos = app.session_state["dataset_preparado"]
    particion = Particionador(
        ConfiguracionParticion(
            porcentaje_test=0.25,
            columna_estratificacion="clase",
            semilla=31,
        )
    ).dividir(datos)
    app.session_state["particion_global"] = particion
    app.run()
    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    app.checkbox(key="clasif_usar_particion_global").check().run()

    app.sidebar.button(key="nav_clasificacion_knn").click().run()
    app.button(key="lab2_entrenar_KNN").click().run()
    knn = app.session_state["resultado_lab2_modelo_KNN"]
    app.sidebar.button(key="nav_clasificacion").click().run()
    app.button(key="lab2_entrenar_RF").click().run()
    rf = app.session_state["resultado_lab2_modelo_RF"]
    assert knn.y_true.index.equals(rf.y_true.index)
    assert knn.y_true.index.equals(particion.test.index)
    assert knn.target == rf.target == "clase"
    assert not app.exception


def test_classifier_route_offers_setup_path_when_dataset_is_not_configurable():
    app = nueva_app()
    app.session_state["dataset_preparado"] = app.session_state["dataset_preparado"][
        ["a"]
    ]
    app.session_state["vista_activa"] = "clasificacion_knn"
    app.run()
    assert not app.exception
    assert any(
        item.label == "Abrir configuración de clasificación" for item in app.button
    )
    app.button(key="abrir_configuracion_clasificacion").click().run()
    assert app.session_state["vista_activa"] == "clasificacion_configuracion"
    assert any("al menos dos columnas" in item.value for item in app.warning)
    assert not app.exception


def test_data_health_summary_is_single_and_tracks_applied_preparation():
    app = nueva_app()
    dataset = app.session_state["dataset_preparado"]
    dataset_with_duplicate = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)
    app.session_state["dataset_original"] = dataset_with_duplicate.copy()
    app.session_state["dataset_preparado"] = dataset_with_duplicate.copy()
    app.run()

    def assert_single_summary(expected_rows):
        active = app.session_state["dataset_preparado"]
        expected = {
            "Filas": expected_rows,
            "Columnas": len(active.columns),
            "Nulos": int(active.isna().sum().sum()),
            "Duplicados": int(active.duplicated().sum()),
        }
        for label, value in expected.items():
            matches = [metric for metric in app.metric if metric.label == label]
            assert len(matches) == 1
            assert matches[0].value == str(value)
        assert not any(
            expander.label == "Resumen del dataset activo" for expander in app.expander
        )

    assert_single_summary(len(dataset_with_duplicate))
    assert any(metric.label == "Outliers Criticos" for metric in app.metric)
    next(
        button for button in app.button if button.label == "Aplicar preparación"
    ).click().run()
    assert_single_summary(len(dataset))
    assert app.session_state["dataset_preparado"].duplicated().sum() == 0

    app.sidebar.button(key="nav_eda").click().run()
    assert any(
        expander.label == "Resumen del dataset activo" for expander in app.expander
    )
    assert any("Outliers críticos" in item.value for item in app.markdown)
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


def test_applying_a_new_global_split_invalidates_read_only_results():
    app = nueva_app()
    ejecutar_comparacion(app)
    app.sidebar.button(key="nav_datos").click().run()
    app.button(key="particion_calcular").click().run()
    assert not app.exception
    assert "resultado_lab2_modelo_KNN" not in app.session_state
    app.sidebar.button(key="nav_lab2_resultados").click().run()
    assert any("Todavía no hay resultados" in e.value for e in app.info)
