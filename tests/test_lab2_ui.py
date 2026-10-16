"""Headless checks of LAB02 navigation, real training and stale-state safety.

All datasets in these tests are synthetic; they are not academic results for
Potabilidad or Diabetes.
"""

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.ui.estado_clasificacion import huella_datos, sincronizar_contexto
from framework_ia.ui.lab2 import _metricas_porcentuales
from framework_ia.utils import configuraciones_experimento

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
TOTAL_CANDIDATOS = sum(len(lista) for lista in configuraciones_experimento().values())


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
    contenido = datos.to_csv(index=False).encode("utf-8")
    app.session_state["fuente_metadata"] = {
        "identidad": "synthetic-ui-fixture",
        "sha256": hashlib.sha256(contenido).hexdigest(),
        "fuente": "fixture sintético para verificación de software",
        "lectura": {"separador": ",", "decimal": ".", "encoding": "utf-8"},
    }
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
    assert any(
        "Todavía no hay ejecuciones individuales" in alerta.value for alerta in app.info
    )


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
    assert app.sidebar.button(key="nav_clasificacion_knn").disabled
    app.run()  # Direct/current routes remain guarded even when navigation is disabled.
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
    assert any("Todavía no hay ejecuciones individuales" in e.value for e in app.info)


@pytest.fixture(scope="module")
def resultado_benchmark():
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=7,
    )
    datos = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    datos["clase"] = y
    return Clasificacion(dataframe=datos, target="clase").experimentar()


def _abrir_experimentos(app, estrategia="holdout"):
    """Abre la vista; las pruebas históricas fijan validación simple explícitamente."""
    app.sidebar.button(key="nav_lab2_benchmark").click().run()
    assert not app.exception
    if app.radio(key="lab2_benchmark_estrategia").value != estrategia:
        app.radio(key="lab2_benchmark_estrategia").set_value(estrategia).run()
        assert not app.exception


def test_benchmark_usa_validacion_cruzada_por_defecto_y_la_ejecuta():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    app.multiselect(key="clasif_features").set_value(["a", "b", "c"]).run()
    app.sidebar.button(key="nav_lab2_benchmark").click().run()
    assert app.radio(key="lab2_benchmark_estrategia").value == "cv"
    assert not any(widget.key == "lab2_benchmark_validacion" for widget in app.slider)
    original = Clasificacion.experimentar
    with patch.object(
        Clasificacion, "experimentar", autospec=True, side_effect=original
    ) as llamada:
        app.button(key="lab2_benchmark_ejecutar").click().run()
    assert not app.exception
    assert llamada.call_count == 1
    assert llamada.call_args.kwargs["estrategia"] == "cv"
    assert llamada.call_args.kwargs["folds"] == 5
    assert "validation_size" not in llamada.call_args.kwargs
    guardado = app.session_state["resultado_lab2_experimento"]
    metadatos = guardado["resultado"].metadatos
    assert metadatos["estrategia_seleccion"] == "validacion_cruzada_estratificada"
    assert metadatos["muestra_train"] + metadatos["muestra_test"] == 120
    assert guardado["argumentos"]["estrategia"] == "cv"
    assert any("validación cruzada de 5 pliegues" in texto.value for texto in app.caption)


def _guardar_benchmark(app, resultado):
    _abrir_experimentos(app)
    with patch.object(Clasificacion, "experimentar", return_value=resultado) as llamada:
        app.button(key="lab2_benchmark_ejecutar").click().run()
    assert not app.exception
    assert llamada.call_count == 1
    return app.session_state["resultado_lab2_experimento"]


def test_benchmark_navigation_catalog_and_shared_setup_do_not_train():
    app = nueva_app()
    with (
        patch.object(
            Clasificacion, "entrenar", side_effect=AssertionError("navigation trained")
        ) as entrenamiento,
        patch.object(
            Clasificacion,
            "experimentar",
            side_effect=AssertionError("navigation experimented"),
        ) as experimento,
    ):
        _abrir_experimentos(app)
        catalogo = next(
            frame.value
            for frame in app.dataframe
            if "Configuración" in frame.value.columns
        )
        assert len(catalogo) == TOTAL_CANDIDATOS
        assert set(catalogo["Algoritmo"]) == {"KNN", "DT", "RF", "XGBoost", "AdaBoost"}
        assert app.selectbox(key="lab2_benchmark_metrica").value == "f1_macro"
        assert app.slider(key="lab2_benchmark_validacion").value == 20
        assert not any(widget.key == "clasif_target" for widget in app.selectbox)
        assert not any(widget.key == "clasif_features" for widget in app.multiselect)
        assert "resultado_lab2_experimento" not in app.session_state
        entrenamiento.assert_not_called()
        experimento.assert_not_called()


def test_benchmark_button_runs_once_with_shared_config_and_exports_real_result():
    app = nueva_app()
    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    app.multiselect(key="clasif_features").set_value(["a", "b", "c"]).run()
    app.number_input(key="clasif_random_state").set_value(19).run()
    _abrir_experimentos(app)
    original = Clasificacion.experimentar
    with patch.object(
        Clasificacion, "experimentar", autospec=True, side_effect=original
    ) as llamada:
        app.button(key="lab2_benchmark_ejecutar").click().run()
    assert not app.exception
    assert llamada.call_count == 1
    assert llamada.call_args.kwargs["validation_size"] == 0.2
    assert llamada.call_args.kwargs["random_state"] == 19
    guardado = app.session_state["resultado_lab2_experimento"]
    resultado = guardado["resultado"]
    assert len(resultado.tabla) == TOTAL_CANDIDATOS
    assert len(resultado.mejores) == 5
    assert resultado.metadatos["muestra_train"] == 72
    assert resultado.metadatos["muestra_validacion"] == 18
    assert resultado.metadatos["muestra_test"] == 30
    assert all(
        ganador.features == ("a", "b", "c") for ganador in resultado.mejores.values()
    )
    validacion = next(
        frame.value
        for frame in app.dataframe
        if "puntuacion_validacion" in frame.value.columns
    )
    assert not any("test" in columna for columna in validacion.columns)
    prueba = next(
        frame.value
        for frame in app.dataframe
        if "Accuracy prueba" in frame.value.columns
    )
    assert len(prueba) == 5
    grafica = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert grafica["layout"]["xaxis"]["range"] == [0, 1]
    assert grafica["data"][0]["orientation"] == "h"
    with ZipFile(BytesIO(guardado["zip"])) as archivo:
        for nombre, contenido in guardado["archivos"].items():
            assert archivo.read(nombre) == contenido
        config = json.loads(archivo.read("configuracion.json"))
        assert (
            config["dataset"]["sha256"]
            == app.session_state["fuente_metadata"]["sha256"]
        )
        assert config["argumentos"]["semilla"] == 19
        assert not config["experimento"]["test_usado_para_seleccion"]
        assert "KNN/predicciones_test.csv" in archivo.namelist()
    with (
        patch.object(
            Clasificacion, "entrenar", side_effect=AssertionError("results trained")
        ),
        patch.object(
            Clasificacion,
            "experimentar",
            side_effect=AssertionError("results experimented"),
        ),
    ):
        app.sidebar.button(key="nav_lab2_resultados").click().run()
        assert not app.exception
        assert any("Accuracy prueba" in frame.value.columns for frame in app.dataframe)
        assert app.sidebar.button(key="nav_lab2_experimentos").disabled
        app.session_state["vista_activa"] = "lab2_experimentos"
        app.run()  # No individual snapshots: direct consultation must remain read-only.
        assert not app.exception
        assert not any(
            "puntuacion_validacion" in frame.value.columns for frame in app.dataframe
        )


@pytest.mark.parametrize("validacion", [0.0, 0.2])
def test_benchmark_reuses_global_validation_or_splits_only_global_train(validacion):
    from framework_ia.datos.particion import Particionador
    from framework_ia.utils import ConfiguracionParticion

    app = nueva_app()
    particion = Particionador(
        ConfiguracionParticion(
            porcentaje_test=0.25,
            porcentaje_validacion=validacion,
            columna_estratificacion="clase",
            semilla=31,
        )
    ).dividir(app.session_state["dataset_preparado"])
    app.session_state["particion_global"] = particion
    app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
    app.checkbox(key="clasif_usar_particion_global").check().run()
    _abrir_experimentos(app)
    assert any(widget.key == "lab2_benchmark_validacion" for widget in app.slider) == (
        validacion == 0.0
    )
    original = Clasificacion.experimentar
    with patch.object(
        Clasificacion, "experimentar", autospec=True, side_effect=original
    ) as llamada:
        app.button(key="lab2_benchmark_ejecutar").click().run()
    assert not app.exception
    resultado = app.session_state["resultado_lab2_experimento"]["resultado"]
    assert set(resultado.metadatos["test_indices"]) == set(particion.test.index)
    assert set(resultado.metadatos["train_indices"]).issubset(
        set(particion.train.index)
    )
    if validacion:
        assert "validation_size" not in llamada.call_args.kwargs
        assert set(resultado.metadatos["validacion_indices"]) == set(
            particion.validacion.index
        )
        assert resultado.metadatos["validacion_externa"]
    else:
        assert llamada.call_args.kwargs["validation_size"] == 0.2
        assert set(resultado.metadatos["validacion_indices"]).issubset(
            set(particion.train.index)
        )
        assert not resultado.metadatos["validacion_externa"]


def test_benchmark_dirty_controls_and_failed_retry_preserve_executed_evidence(
    resultado_benchmark,
):
    app = nueva_app()
    guardado = _guardar_benchmark(app, resultado_benchmark)
    evidencia = guardado["zip"]
    with patch.object(
        Clasificacion, "experimentar", side_effect=AssertionError("widget trained")
    ):
        app.selectbox(key="lab2_benchmark_metrica").set_value("accuracy").run()
    assert app.session_state["resultado_lab2_experimento"] is guardado
    assert any("Las opciones cambiaron" in aviso.value for aviso in app.info)
    with patch.object(
        Clasificacion, "experimentar", side_effect=ValueError("fallo sintético")
    ):
        app.button(key="lab2_benchmark_ejecutar").click().run()
    assert not app.exception
    assert app.session_state["resultado_lab2_experimento"] is guardado
    assert app.session_state["resultado_lab2_experimento"]["zip"] == evidencia
    assert any("fallo sintético" in aviso.value for aviso in app.error)
    assert guardado["opciones"]["metrica"] == "f1_macro"
    assert (
        json.loads(guardado["archivos"]["configuracion.json"])["argumentos"]["metrica"]
        == "f1_macro"
    )


def test_benchmark_partial_candidate_failures_are_visible_and_exported(
    resultado_benchmark,
):
    tabla = resultado_benchmark.tabla.copy()
    afectados = tabla.algoritmo == "XGBoost"
    tabla.loc[afectados, ["estado", "error", "seleccionado"]] = [
        "error",
        "XGBoost no instalado",
        False,
    ]
    parcial = replace(
        resultado_benchmark,
        tabla=tabla,
        mejores={
            codigo: ganador
            for codigo, ganador in resultado_benchmark.mejores.items()
            if codigo != "XGBoost"
        },
        metadatos={
            **resultado_benchmark.metadatos,
            "candidatos_fallidos": 3,
            "errores_prueba": {"XGBoost": "sin modelo"},
        },
    )
    app = nueva_app()
    guardado = _guardar_benchmark(app, parcial)
    assert any("Ejecución parcial" in aviso.value for aviso in app.warning)
    assert any("XGBoost" in aviso.value for aviso in app.error)
    exportado = pd.read_csv(BytesIO(guardado["archivos"]["comparacion_validacion.csv"]))
    assert (exportado.loc[exportado.algoritmo == "XGBoost", "estado"] == "error").all()
    assert "XGBoost/predicciones_test.csv" not in guardado["archivos"]


def test_benchmark_no_winners_keeps_previous_success_and_shows_attempt_errors(
    resultado_benchmark,
):
    app = nueva_app()
    guardado = _guardar_benchmark(app, resultado_benchmark)
    tabla = resultado_benchmark.tabla.copy()
    tabla["estado"] = "error"
    tabla["error"] = "todos fallaron"
    tabla["seleccionado"] = False
    fallido = replace(
        resultado_benchmark, tabla=tabla, mejores={}, mejor_algoritmo=None
    )
    with patch.object(Clasificacion, "experimentar", return_value=fallido):
        app.button(key="lab2_benchmark_ejecutar").click().run()
    assert not app.exception
    assert app.session_state["resultado_lab2_experimento"] is guardado
    assert (
        app.session_state["resultado_lab2_intento_experimento"]["resultado"] is fallido
    )
    assert any("no produjo ganadores" in aviso.value for aviso in app.warning)
    assert any("todos fallaron" in str(frame.value) for frame in app.dataframe)


@pytest.mark.parametrize("cambio", ["semilla", "particion", "dataset"])
def test_benchmark_context_change_removes_saved_download_and_error(
    cambio, resultado_benchmark
):
    app = nueva_app()
    _guardar_benchmark(app, resultado_benchmark)
    app.session_state["error_lab2_experimento"] = "error viejo"
    if cambio == "dataset":
        datos = app.session_state["dataset_original"].copy()
        datos["a"] += 100
        app.session_state["fuente_aplicada"] = (datos, "otro.csv", "otro-csv")
        app.run()
    else:
        app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
        if cambio == "semilla":
            app.number_input(key="clasif_random_state").set_value(99).run()
        else:
            app.slider(key="clasif_test_size").set_value(30).run()
    assert not app.exception
    assert "resultado_lab2_experimento" not in app.session_state
    assert "error_lab2_experimento" not in app.session_state
    _abrir_experimentos(app)
    assert not app.get("download_button")


def test_benchmark_global_imputation_blocks_execution():
    app = nueva_app()
    app.session_state["dataset_preparacion"] = {"imputar": True, "escalado": "Ninguno"}
    with patch.object(
        Clasificacion, "experimentar", side_effect=AssertionError("unsafe experiment")
    ):
        _abrir_experimentos(app)
    assert app.button(key="lab2_benchmark_ejecutar").disabled
    assert any("restaure el dataset original" in aviso.value for aviso in app.error)


def test_applied_csv_provenance_tracks_exact_bytes_and_read_options(tmp_path):
    from framework_ia.ui import streamlit_app
    from framework_ia.utils import ConfiguracionCSV

    primero = tmp_path / "primero.csv"
    segundo = tmp_path / "segundo.csv"
    primero.write_bytes(b"x;clase\n1,5;0\n2,5;1\n")
    segundo.write_bytes(b"x;clase\n9,5;0\n8,5;1\n")
    estado = {}
    lectura = ConfiguracionCSV(separador=";", decimal=",", encoding="utf-8")
    with patch.object(streamlit_app.st, "session_state", estado):
        streamlit_app._cargar_fuente_csv(
            primero, lectura, fuente="referencia del profesor"
        )
        metadata = estado["fuente_metadata"].copy()
        assert estado["fuente_aplicada"][0].x.iloc[0] == 1.5
        assert metadata["sha256"] == hashlib.sha256(primero.read_bytes()).hexdigest()
        assert metadata["fuente"] == "referencia del profesor"
        assert metadata["lectura"]["separador"] == ";"
        streamlit_app._cargar_fuente_csv(segundo, lectura)
    assert (
        estado["fuente_metadata"]["sha256"]
        == hashlib.sha256(segundo.read_bytes()).hexdigest()
    )
    assert estado["fuente_metadata"]["sha256"] != metadata["sha256"]
    assert estado["fuente_metadata"]["fuente"] is None


def test_local_source_lists_nested_csvs_after_root_example_and_applies_read_options(
    tmp_path,
):
    from framework_ia.ui import streamlit_app

    data = tmp_path / "data"
    nested = data / "lab02"
    nested.mkdir(parents=True)
    ejemplo = data / "ejemplo.csv"
    ejemplo.write_bytes(b"x,clase\n1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n")
    diabetes = nested / "Diabetes.csv"
    diabetes.write_bytes(b"x;clase\n1,5;0\n2,5;1\n3,5;0\n4,5;1\n5,5;0\n6,5;1\n")
    potabilidad = nested / "Potabilidad.csv"
    potabilidad.write_bytes(b"x,clase\n2,0\n4,1\n6,0\n8,1\n10,0\n12,1\n")
    with patch.object(streamlit_app, "BASE_DIR", tmp_path):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=60).run()
        assert not app.exception
        opciones = app.selectbox(key="fuente_archivo").options
        assert opciones == [
            "ejemplo.csv",
            str(Path("lab02") / "Diabetes.csv"),
            str(Path("lab02") / "Potabilidad.csv"),
        ]
        assert app.session_state["fuente_aplicada"][1] == "ejemplo.csv"
        app.selectbox(key="fuente_archivo").select_index(1)
        app.text_input(key="fuente_separador").set_value(";")
        app.text_input(key="fuente_decimal").set_value(",")
        app.text_input(key="fuente_referencia").set_value(
            "CSV entregado por el profesor"
        )
        next(
            button for button in app.button if button.label == "Aplicar fuente"
        ).click().run()
        assert not app.exception
        assert app.session_state["dataset_original"].x.iloc[0] == 1.5
        assert (
            app.session_state["fuente_metadata"]["sha256"]
            == hashlib.sha256(diabetes.read_bytes()).hexdigest()
        )
        assert (
            app.session_state["fuente_metadata"]["fuente"]
            == "CSV entregado por el profesor"
        )
        assert app.session_state["fuente_metadata"]["lectura"]["decimal"] == ","


def test_reapplied_local_csv_with_preserved_timestamp_updates_data_and_evidence(
    tmp_path,
):
    import os

    from framework_ia.ui import streamlit_app
    from framework_ia.utils import ConfiguracionCSV

    csv = tmp_path / "mismo.csv"
    csv.write_bytes(b"x,clase\n1,0\n2,1\n")
    fecha = csv.stat()
    estado = {}
    with patch.object(streamlit_app.st, "session_state", estado):
        streamlit_app._cargar_fuente_csv(csv, ConfiguracionCSV())
        datos, _, identidad = estado["fuente_aplicada"]
        streamlit_app._sincronizar_dataset(datos, identidad)
        estado["resultado_lab2_experimento"] = {"evidencia": "anterior"}
        csv.write_bytes(b"x,clase\n9,0\n8,1\n")
        os.utime(csv, ns=(fecha.st_atime_ns, fecha.st_mtime_ns))
        assert csv.stat().st_mtime_ns == fecha.st_mtime_ns
        streamlit_app._cargar_fuente_csv(csv, ConfiguracionCSV())
        datos_nuevos, _, identidad_nueva = estado["fuente_aplicada"]
        streamlit_app._sincronizar_dataset(datos_nuevos, identidad_nueva)
    assert identidad_nueva != identidad
    assert estado["dataset_preparado"].x.tolist() == [9, 8]
    assert (
        estado["fuente_metadata"]["sha256"]
        == hashlib.sha256(csv.read_bytes()).hexdigest()
    )
    assert "resultado_lab2_experimento" not in estado


def _fuentes_diabetes_sinteticas(tmp_path):
    from framework_ia.datos.diabetes_lab02 import PREDICTORAS_DIABETES

    filas = 60
    datos = pd.DataFrame(
        {
            columna: [float(i + indice + 1) for i in range(filas)]
            for indice, columna in enumerate(PREDICTORAS_DIABETES)
        }
    )
    datos["location"] = ["Louisa", "Buckingham"] * (filas // 2)
    datos["gender"] = ["female", "male"] * (filas // 2)
    datos["frame"] = ["small", "medium", "large"] * (filas // 3)
    datos.insert(0, "id", range(1, filas + 1))
    datos["glyhb"] = [5.0, 8.0] * (filas // 2)
    datos["diabetes"] = ["No diabetes", "Diabetes"] * (filas // 2)
    original = datos.drop(columns="diabetes")
    preparado_sha = hashlib.sha256(datos.to_csv(index=False).encode()).hexdigest()
    original_sha = hashlib.sha256(original.to_csv(index=False).encode()).hexdigest()
    procedencia = tmp_path / "procedencia.json"
    procedencia.write_text(
        json.dumps(
            {
                "diabetes": {
                    "sha256": preparado_sha,
                    "original": {"sha256": original_sha},
                    "derivacion": {
                        "regla": {"positiva": "Diabetes", "negativa": "No diabetes"}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return datos, original, preparado_sha, original_sha, procedencia


def _app_fuente_diabetes(datos, sha, *, identidad="diabetes-fixture"):
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (datos, "diabetes.csv", identidad)
    app.session_state["fuente_metadata"] = {
        "identidad": identidad,
        "sha256": sha,
        "fuente": "fixture sintético de configuración; sin resultados académicos",
        "lectura": {"separador": ",", "decimal": ".", "encoding": "utf-8"},
    }
    app.run()
    assert not app.exception
    return app


def test_diabetes_preset_applies_native_features_before_widgets_and_never_trains(
    tmp_path,
):
    from framework_ia.datos.diabetes_lab02 import PREDICTORAS_DIABETES
    from framework_ia.datos.particion import Particionador
    from framework_ia.ui import perfil_diabetes
    from framework_ia.utils import ConfiguracionParticion

    datos, _, digest, _, procedencia = _fuentes_diabetes_sinteticas(tmp_path)
    with patch.object(perfil_diabetes, "RUTA_PROCEDENCIA", procedencia):
        app = _app_fuente_diabetes(datos, digest)
        assert app.sidebar.button(key="nav_lab2_benchmark").disabled
        app.session_state["lab2_benchmark_validacion"] = 30
        app.session_state["lab2_benchmark_metrica"] = "accuracy"
        particion = Particionador(
            ConfiguracionParticion(
                porcentaje_test=0.3, columna_estratificacion="diabetes", semilla=9
            )
        ).dividir(datos)
        app.session_state["particion_global"] = particion
        app.session_state["clasif_target"] = "glyhb"
        app.session_state["clasif_features"] = ["id"]
        app.session_state["clasif_random_state"] = 99
        app.session_state["clasif_test_size"] = 40
        app.session_state["clasif_usar_particion_global"] = True
        app.sidebar.button(key="nav_clasificacion_configuracion").click().run()
        assert not app.exception
        app.session_state["resultado_lab2_experimento"] = {"zip": b"evidencia anterior"}
        app.session_state["resultado_lab2_modelo_KNN"] = {"anterior": True}
        app.session_state["error_lab2_experimento"] = "error anterior"
        with (
            patch.object(Clasificacion, "entrenar") as entrenar,
            patch.object(Clasificacion, "experimentar") as experimentar,
        ):
            app.button(key="aplicar_configuracion_diabetes_lab02").click().run()
            assert not app.exception
            assert app.selectbox(key="clasif_target").value == "diabetes"
            assert app.multiselect(key="clasif_features").value == list(
                PREDICTORAS_DIABETES
            )
            assert len(app.multiselect(key="clasif_features").value) == 17
            assert not {"id", "glyhb"} & set(
                app.multiselect(key="clasif_features").value
            )
            assert app.checkbox(key="clasif_incluir_categoricas").value
            assert app.checkbox(key="clasif_imputar").value
            assert app.checkbox(key="clasif_escalar").value
            assert app.checkbox(key="clasif_strat").value
            assert not app.checkbox(key="clasif_usar_particion_global").value
            assert app.number_input(key="clasif_random_state").value == 42
            assert app.slider(key="clasif_test_size").value == 25
            assert "resultado_lab2_experimento" not in app.session_state
            assert "resultado_lab2_modelo_KNN" not in app.session_state
            assert "error_lab2_experimento" not in app.session_state
            _abrir_experimentos(app)
            assert app.slider(key="lab2_benchmark_validacion").value == 20
            assert app.selectbox(key="lab2_benchmark_metrica").value == "f1_macro"
            assert not app.button(key="lab2_benchmark_ejecutar").disabled
            entrenar.assert_not_called()
            experimentar.assert_not_called()


def test_original_diabetes_is_blocked_until_the_prepared_source_is_applied(tmp_path):
    from framework_ia.ui import perfil_diabetes

    datos, original, preparado_sha, original_sha, procedencia = (
        _fuentes_diabetes_sinteticas(tmp_path)
    )
    with patch.object(perfil_diabetes, "RUTA_PROCEDENCIA", procedencia):
        app = _app_fuente_diabetes(original, original_sha)
        with patch.object(Clasificacion, "experimentar") as experimentar:
            assert app.sidebar.button(key="nav_lab2_benchmark").disabled
            app.session_state["vista_activa"] = "lab2_benchmark"
            app.run()
            assert app.button(key="lab2_benchmark_ejecutar").disabled
            assert any(
                "CSV original de Diabetes" in aviso.value for aviso in app.warning
            )
            assert not any(
                button.key == "aplicar_configuracion_diabetes_lab02"
                for button in app.button
            )
            with pytest.raises(AppTestError, match="Cannot update a disabled button"):
                app.button(key="lab2_benchmark_ejecutar").click()
            experimentar.assert_not_called()
        app.session_state["fuente_aplicada"] = (
            datos,
            "diabetes.csv",
            "diabetes-preparado",
        )
        app.session_state["fuente_metadata"] = {
            "identidad": "diabetes-preparado",
            "sha256": preparado_sha,
        }
        app.run()
        assert not app.exception
        assert any(
            button.key == "aplicar_configuracion_diabetes_lab02"
            for button in app.button
        )
        assert app.button(key="lab2_benchmark_ejecutar").disabled
        app.button(key="aplicar_configuracion_diabetes_lab02").click().run()
        assert not app.button(key="lab2_benchmark_ejecutar").disabled
        assert not any(
            "CSV original de Diabetes" in aviso.value for aviso in app.warning
        )


def test_diabetes_preset_does_not_match_generic_same_filename(tmp_path):
    from framework_ia.ui import perfil_diabetes

    datos, _, _, _, procedencia = _fuentes_diabetes_sinteticas(tmp_path)
    with patch.object(perfil_diabetes, "RUTA_PROCEDENCIA", procedencia):
        app = _app_fuente_diabetes(datos, "hash-de-otro-csv")
        assert app.sidebar.button(key="nav_lab2_benchmark").disabled
        app.session_state["vista_activa"] = "lab2_benchmark"
        app.run()
        assert not any(
            button.key == "aplicar_configuracion_diabetes_lab02"
            for button in app.button
        )
        assert app.button(key="lab2_benchmark_ejecutar").disabled
        assert any("categóricas" in aviso.value for aviso in app.warning)


def test_diabetes_profile_exact_sha_requires_matching_data_and_safe_preparation(
    tmp_path,
):
    from framework_ia.ui import perfil_diabetes

    datos, _, preparado_sha, original_sha, procedencia = _fuentes_diabetes_sinteticas(
        tmp_path
    )
    with patch.object(perfil_diabetes, "RUTA_PROCEDENCIA", procedencia):
        metadata = {"identidad": "fixture", "sha256": preparado_sha}
        estado = {
            "fuente_aplicada": (datos, "diabetes.csv", "fixture"),
            "fuente_metadata": metadata,
        }
        assert perfil_diabetes.estado_fuente_diabetes(datos, estado) == "preparado"
        modificados = datos.copy()
        modificados.loc[0, "age"] += 1
        assert perfil_diabetes.estado_fuente_diabetes(modificados, estado) is None
        assert (
            perfil_diabetes.estado_fuente_diabetes(
                datos, {**estado, "fuente_metadata": {**metadata, "identidad": "vieja"}}
            )
            is None
        )
        assert (
            perfil_diabetes.estado_fuente_diabetes(
                datos, {**estado, "dataset_preparacion": {"imputar": True}}
            )
            is None
        )
        assert (
            perfil_diabetes.estado_fuente_diabetes(
                datos,
                {**estado, "fuente_metadata": {**metadata, "sha256": original_sha}},
            )
            == "original"
        )
        assert (
            perfil_diabetes.estado_fuente_diabetes(
                datos, {**estado, "fuente_metadata": {**metadata, "sha256": "otro-csv"}}
            )
            is None
        )


@pytest.mark.parametrize(
    "seccion",
    [
        {"original": None},
        {"original": []},
        {"derivacion": None},
        {"derivacion": []},
        {"derivacion": {"regla": None}},
        {"derivacion": {"regla": []}},
    ],
)
def test_diabetes_profile_malformed_nested_json_keeps_generic_data_operable(
    tmp_path, seccion
):
    from framework_ia.ui import perfil_diabetes

    datos, _, digest, _, procedencia = _fuentes_diabetes_sinteticas(tmp_path)
    contenido = json.loads(procedencia.read_text(encoding="utf-8"))
    contenido["diabetes"].update(seccion)
    procedencia.write_text(json.dumps(contenido), encoding="utf-8")
    estado = {
        "fuente_aplicada": (datos, "diabetes.csv", "fixture"),
        "fuente_metadata": {"identidad": "fixture", "sha256": digest},
    }
    with patch.object(perfil_diabetes, "RUTA_PROCEDENCIA", procedencia):
        assert perfil_diabetes.estado_fuente_diabetes(datos, estado) is None
