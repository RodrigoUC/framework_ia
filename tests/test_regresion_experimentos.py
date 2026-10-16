"""Regression benchmark checks use synthetic fixtures, not academic results."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_regression

from framework_ia import Regresion
from framework_ia.datos.particion import Particionador
from framework_ia.modelos.supervisado import experimentos_regresion as experimentos
from framework_ia.modelos.supervisado.preprocesamiento import PreprocesadorSupervisado
from framework_ia.resultados import ResultadoExperimentoRegresion
from framework_ia.utils import ALGORITMOS_LAB3, ConfiguracionParticion


@pytest.fixture
def data():
    X, y = make_regression(n_samples=80, n_features=3, noise=4, random_state=71)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["target"] = y
    frame.index = pd.Index([f"row-{i}" for i in range(len(frame))], name="id")
    return frame


def split(data, validation=0):
    return Particionador(
        ConfiguracionParticion(
            porcentaje_test=0.25, porcentaje_validacion=validation, semilla=19
        )
    ).dividir(data)


def candidate(name, **parameters):
    return {"nombre": name, "parametros": parameters}


def test_default_catalog_all_nine_models_fourteen_candidates_and_reproducibility(data):
    model = Regresion(dataframe=data, target="target")
    result = model.experimentar(feature_simple="b")
    assert isinstance(result, ResultadoExperimentoRegresion)
    assert len(result.tabla) == 14
    assert result.tabla.estado.eq("ok").all()
    assert tuple(result.mejores) == ALGORITMOS_LAB3
    assert result.metadatos["feature_simple"] == "b"
    assert result.mejores["RLS"].features == ("b",)
    assert result.mejores["RLM"].features == ("a", "b", "c")
    assert model.features == ["a", "b", "c"]
    assert model.experimento is result
    assert model.evaluar() is result.mejores[result.mejor_algoritmo]
    assert Regresion.benchmark is Regresion.experimentar
    repeated = Regresion(dataframe=data).benchmark(feature_simple="b")
    assert_frame_equal(result.tabla, repeated.tabla)
    for _, group in result.tabla.groupby("algoritmo", sort=False):
        selected = group[group.seleccionado]
        assert len(selected) == 1
        assert (
            selected.puntuacion_validacion.iloc[0] == group.puntuacion_validacion.min()
        )
    selected_rows = result.tabla[result.tabla.seleccionado]
    assert (
        result.mejor_algoritmo
        == selected_rows.loc[selected_rows.puntuacion_validacion.idxmin(), "algoritmo"]
    )
    for name in ("LassoCV", "RidgeCV"):
        assert isinstance(result.mejores[name].parametros["cv"], dict)
        assert result.mejores[name].metadatos["alpha_elegido"] > 0


def test_common_outer_folds_fit_preprocessing_only_on_their_training_rows(data):
    partition = split(data)
    observed = []
    original = PreprocesadorSupervisado.fit

    def fit(preprocessor, X, y=None):
        fitted = original(preprocessor, X, y)
        scaler = preprocessor.transformador_.named_transformers_[
            "numericas"
        ].named_steps["escalar"]
        np.testing.assert_allclose(scaler.mean_, X.mean().to_numpy())
        observed.append(X.index.tolist())
        return fitted

    with patch.object(PreprocesadorSupervisado, "fit", autospec=True, side_effect=fit):
        result = Regresion(dataframe=data).experimentar(
            algoritmos=["Ridge"],
            configuraciones={"Ridge": [candidate("one", alpha=1)]},
            particion=partition,
            folds=3,
        )
    metadata = result.metadatos
    assert observed[:-1] == [fold["train_indices"] for fold in metadata["pliegues"]]
    assert observed[-1] == partition.train.index.tolist()
    assert all(not set(indices) & set(partition.test.index) for indices in observed)
    validation = metadata["pliegues_indices"]
    assert set().union(*map(set, validation)) == set(partition.train.index)
    assert sum(map(len, validation)) == len(partition.train)
    row = result.tabla.iloc[0]
    assert row.puntuacion_validacion == np.mean(row.puntuaciones_pliegues)
    assert row.puntuacion_validacion_std == np.std(row.puntuaciones_pliegues, ddof=1)


@pytest.mark.parametrize("strategy", [None, "cv"])
def test_external_validation_default_holdout_or_reserved_under_explicit_cv(
    data, strategy
):
    partition = split(data, validation=0.2)
    result = Regresion(dataframe=data).experimentar(
        algoritmos=["RLS", "Ridge"], particion=partition, estrategia=strategy, folds=3
    )
    metadata = result.metadatos
    assert metadata["train_indices"] == partition.train.index.tolist()
    assert metadata["test_indices"] == partition.test.index.tolist()
    assert metadata["validacion_indices"] == partition.validacion.index.tolist()
    assert metadata["estrategia_seleccion"] == ("holdout" if strategy is None else "cv")
    assert metadata["validacion_reservada_indices"] == (
        partition.validacion.index.tolist() if strategy == "cv" else []
    )
    for winner in result.mejores.values():
        assert winner.y_train.index.equals(partition.train.index)
        assert winner.y_true.index.equals(partition.test.index)
    if strategy == "cv":
        assert all(
            not set(fold) & set(partition.validacion.index)
            for fold in metadata["pliegues_indices"]
        )


def test_holdout_without_external_validation_splits_only_shared_train(data):
    partition = split(data)
    result = Regresion(dataframe=data).experimentar(
        algoritmos="Ridge",
        particion=partition,
        estrategia="holdout",
        validation_size=0.2,
    )
    train = set(result.metadatos["train_indices"])
    validation = set(result.metadatos["validacion_indices"])
    assert not train & validation
    assert train | validation == set(partition.train.index)
    assert result.metadatos["test_indices"] == partition.test.index.tolist()
    assert not result.metadatos["reajustado_train_completo"]
    assert result.tabla.puntuacion_validacion_std.eq(0).all()


@pytest.mark.parametrize("strategy", [None, "cv"])
def test_reserved_values_cannot_change_validation_or_selection(data, strategy):
    partition = split(data, validation=0.2)
    options = {
        "algoritmos": ["Ridge", "DT"],
        "particion": partition,
        "estrategia": strategy,
        "folds": 3,
    }
    first = Regresion(dataframe=data).experimentar(**options)
    changed = data.copy()
    changed.loc[partition.test.index, "a"] += 10000
    changed.loc[partition.test.index, "target"] += 10000
    validation = partition.validacion
    if strategy == "cv":
        changed.loc[partition.validacion.index, ["a", "target"]] += 10000
        validation = changed.loc[partition.validacion.index].copy()
    options["particion"] = replace(
        partition, test=changed.loc[partition.test.index].copy(), validacion=validation
    )
    second = Regresion(dataframe=changed).experimentar(**options)
    assert_frame_equal(first.tabla, second.tabla)
    assert first.mejor_algoritmo == second.mejor_algoritmo
    assert first.mejores["Ridge"].metricas != second.mejores["Ridge"].metricas


def test_all_validation_decisions_precede_every_test_evaluation(data):
    events = []
    common_splits = []
    original_validation = experimentos._evaluar
    original_result = Regresion._resultado

    def validation(*args, **kwargs):
        events.append("validation")
        common_splits.append(id(args[2]))
        return original_validation(*args, **kwargs)

    def evaluate(*args, **kwargs):
        events.append("test")
        return original_result(*args, **kwargs)

    with (
        patch.object(experimentos, "_evaluar", side_effect=validation),
        patch.object(Regresion, "_resultado", autospec=True, side_effect=evaluate),
    ):
        Regresion(dataframe=data).experimentar(algoritmos=["Ridge", "DT"], folds=3)
    assert events == ["validation"] * 4 + ["test"] * 2
    assert len(set(common_splits)) == 1


def test_ties_preserve_candidate_and_algorithm_declaration_order(data):
    fixed = {"rmse": 1.0, "mae": 1.0, "mse": 1.0, "r2": 0.0}
    configs = {
        name: [candidate("first"), candidate("second")] for name in ("DT", "Ridge")
    }
    with patch.object(experimentos, "metricas_regresion", return_value=fixed):
        result = Regresion(dataframe=data).experimentar(
            algoritmos=["DT", "Ridge"], configuraciones=configs, folds=3
        )
    assert result.mejor_algoritmo == "DT"
    assert result.metadatos["selecciones"] == {"DT": "first", "Ridge": "first"}


def test_candidate_failures_continue_and_all_failed_retry_preserves_success(data):
    model = Regresion(dataframe=data)
    first = model.experimentar(algoritmos=["DT", "Ridge"], folds=3)
    saved_result, saved_model = model.resultado, model.modelo
    failed = model.experimentar(
        algoritmos="DT",
        folds=3,
        configuraciones={"DT": [candidate("invalid", max_depth=0)]},
    )
    assert not failed.mejores
    assert failed.mejor_algoritmo is None
    assert failed.tabla.estado.eq("error").all()
    assert failed.tabla.error.str.contains("max_depth").all()
    assert model.experimento is first
    assert model.resultado is saved_result and model.modelo is saved_model
    partial = model.experimentar(
        algoritmos=["DT", "Ridge"],
        folds=3,
        configuraciones={"DT": [candidate("invalid", max_depth=0)]},
    )
    assert set(partial.mejores) == {"Ridge"}
    assert partial.metadatos["candidatos_fallidos"] == 1
    assert model.experimento is partial


def test_test_evaluation_failures_are_visible_without_reselection(data):
    model = Regresion(dataframe=data)
    with patch.object(
        Regresion, "_resultado", side_effect=RuntimeError("test unavailable")
    ):
        result = model.experimentar(algoritmos="Ridge", folds=3)
    assert not result.mejores
    assert result.mejor_algoritmo == "Ridge"
    assert "test unavailable" in result.metadatos["errores_prueba"]["Ridge"]
    assert result.tabla.seleccionado.sum() == 1
    assert model.experimento is None


@pytest.mark.parametrize(
    "invalid",
    [
        "target",
        "split",
        "folds",
        "simple",
        "catalog",
        "test_size",
        "training_constant",
    ],
)
def test_invalid_data_splits_or_configuration_abort_before_fitting(data, invalid):
    options = {"algoritmos": "Ridge"}
    if invalid == "target":
        data.loc[data.index[0], "target"] = np.nan
    elif invalid == "split":
        partition = split(data)
        options["particion"] = replace(partition, train=partition.train.iloc[1:])
    elif invalid == "folds":
        options["folds"] = 50
    elif invalid == "simple":
        options["feature_simple"] = "missing"
    elif invalid == "test_size":
        options["test_size"] = 0.001
    elif invalid == "training_constant":
        partition = split(data)
        data.loc[partition.train.index, "target"] = 1.0
        options["particion"] = replace(
            partition, train=data.loc[partition.train.index].copy()
        )
    else:
        options["configuraciones"] = {"Ridge": [candidate("same"), candidate("same")]}
    with (
        patch.object(Regresion, "_ajustar", side_effect=AssertionError("must not fit")),
        pytest.raises((ValueError, KeyError)),
    ):
        Regresion(dataframe=data).experimentar(**options)


def test_single_predictor_default_is_schema_order_not_target_fit(data):
    result = Regresion(dataframe=data, features=["c", "a"]).experimentar(
        algoritmos=["RLS", "RLM"], folds=3
    )
    assert result.metadatos["feature_simple"] == "c"
    assert result.mejores["RLS"].features == ("c",)
    assert result.mejores["RLM"].features == ("c", "a")


def test_nonfinite_candidate_scores_are_failures_not_winners(data):
    invalid = {"rmse": np.inf, "mae": 1.0, "mse": np.inf, "r2": 0.0}
    with patch.object(experimentos, "metricas_regresion", return_value=invalid):
        result = Regresion(dataframe=data).experimentar(algoritmos="Ridge", folds=3)
    assert not result.mejores
    assert result.tabla.estado.eq("error").all()
    assert result.tabla.error.str.contains("no finitas").all()
