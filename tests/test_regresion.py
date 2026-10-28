"""Regression core contracts and train-only preprocessing."""

import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from framework_ia import Regresion
from framework_ia.modelos.supervisado import Supervisado
from framework_ia.resultados import ResultadoParticion, ResultadoRegresion


def regression_data(n=40):
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, n))
    return pd.DataFrame(
        {"a": a, "b": b, "target": 3 * a - 2 * b + 7},
        index=pd.Index([f"row-{i}" for i in range(n)], name="id"),
    )


def partition(data, train, test, validation=()):
    return ResultadoParticion(
        train=data.iloc[list(train)].copy(),
        test=data.iloc[list(test)].copy(),
        validacion=data.iloc[list(validation)].copy() if len(validation) else None,
        porcentaje_train=len(train) / len(data),
        porcentaje_test=len(test) / len(data),
        porcentaje_validacion=len(validation) / len(data),
        columna_estratificacion=None,
        distribucion_train=None,
        distribucion_test=None,
        distribucion_validacion=None,
        semilla=19,
    )


class RegresionTests(unittest.TestCase):
    def setUp(self):
        self.data = regression_data()

    def test_inheritance_aliases_default_target_and_frozen_result(self):
        self.assertTrue(issubclass(Regresion, Supervisado))
        for method in ("lineal_multiple", "RLM", "entrenar"):
            with self.subTest(method=method):
                model = Regresion(dataframe=self.data)
                result = getattr(model, method)()
                self.assertIsInstance(result, ResultadoRegresion)
                self.assertEqual(result.target, "target")
                self.assertEqual(result.features, ("a", "b"))
                self.assertEqual(result.algoritmo, "RLM")
                self.assertEqual((result.muestra_train, result.muestra_test), (30, 10))
                self.assertLess(result.metricas["rmse"], 1e-10)
                self.assertAlmostEqual(model.r2, 1)
                self.assertIs(model.evaluar(), result)
                assert_series_equal(model.predecir(), result.y_pred)
                self.assertIn("R²:", str(model))
                with self.assertRaises(FrozenInstanceError):
                    result.algoritmo = "changed"
        for method in ("RLS", "lineal_simple"):
            with self.subTest(method=method):
                result = getattr(
                    Regresion(dataframe=self.data, features=["a"]), method
                )()
                self.assertEqual(result.algoritmo, "RLS")

    def test_metrics_and_residuals_use_original_target_units(self):
        result = Regresion(dataframe=self.data, features=["a"]).RLS()
        for actual, predicted, metrics, residuals in (
            (result.y_true, result.y_pred, result.metricas, result.residuos),
            (
                result.y_train,
                result.y_pred_train,
                result.metricas_train,
                result.residuos_train,
            ),
        ):
            self.assertEqual(set(metrics), {"rmse", "mae", "mse", "r2"})
            self.assertAlmostEqual(
                metrics["mae"], mean_absolute_error(actual, predicted)
            )
            self.assertAlmostEqual(
                metrics["mse"], mean_squared_error(actual, predicted)
            )
            self.assertAlmostEqual(metrics["rmse"] ** 2, metrics["mse"])
            self.assertAlmostEqual(metrics["r2"], r2_score(actual, predicted))
            assert_series_equal(residuals, (actual - predicted).rename("residuo"))

    def test_prediction_requires_fit_and_aligns_named_columns(self):
        model = Regresion(dataframe=self.data)
        for operation in (model.predecir, model.evaluar):
            with self.assertRaisesRegex(RuntimeError, "Entrene"):
                operation()
        result = model.RLM()
        X = self.data.loc[result.y_true.index]
        assert_series_equal(model.predecir(X[["b", "a"]]), result.y_pred)
        np.testing.assert_allclose(
            model.predecir(X[["a", "b"]].to_numpy()), result.y_pred
        )
        with self.assertRaisesRegex(ValueError, "Faltan variables"):
            model.predecir(X[["a"]])
        with self.assertRaises(TypeError):
            model.predecir(X.iloc[0])
        with self.assertRaisesRegex(ValueError, "únicos"):
            model.predecir(pd.concat([X[["a", "b"]], X[["a"]]], axis=1))

    def test_rejects_invalid_targets_without_silently_dropping_rows(self):
        invalid_targets = (
            ["value"] * 40,
            np.full(40, 1.0),
            [True, False] * 20,
            np.arange(40).astype(complex),
            np.r_[np.nan, np.arange(39)],
            np.r_[np.inf, np.arange(39)],
            np.r_[-np.inf, np.arange(39)],
        )
        for target in invalid_targets:
            with (
                self.subTest(target=str(target[:2])),
                self.assertRaises((ValueError, TypeError)),
            ):
                Regresion(dataframe=self.data.assign(target=target)).RLM()
        nullable = self.data.copy()
        nullable["target"] = pd.array(range(40), dtype="Int64")
        self.assertEqual(Regresion(dataframe=nullable).RLM().muestra_train, 30)
        nullable.iloc[0, -1] = pd.NA
        with self.assertRaisesRegex(ValueError, "nulos"):
            Regresion(dataframe=nullable).RLM()

    def test_validates_columns_indices_and_predictor_contracts(self):
        for features in ([], ["a", "a"], ["a", "target"], ["missing"]):
            with (
                self.subTest(features=features),
                self.assertRaises((ValueError, KeyError)),
            ):
                Regresion(dataframe=self.data, features=features).RLM()
        with self.assertRaises(KeyError):
            Regresion(dataframe=self.data, target="missing").RLM()
        with self.assertRaisesRegex(ValueError, "No hay datos"):
            Regresion(dataframe=pd.DataFrame()).RLM()
        duplicate_indices = self.data.copy()
        duplicate_indices.index = [0] * 40
        with self.assertRaisesRegex(ValueError, "índices únicos"):
            Regresion(dataframe=duplicate_indices).RLM()
        with self.assertRaisesRegex(ValueError, "columna únicos"):
            Regresion(dataframe=pd.concat([self.data, self.data[["a"]]], axis=1)).RLM()
        with self.assertRaisesRegex(ValueError, "exactamente una"):
            Regresion(dataframe=self.data).RLS()
        with self.assertRaisesRegex(ValueError, "al menos dos"):
            Regresion(dataframe=self.data, features=["a"]).RLM()
        with self.assertRaisesRegex(TypeError, "numérica real"):
            Regresion(
                dataframe=self.data.assign(a=["a", "b"] * 20), features=["a"]
            ).RLS()
        with self.assertRaisesRegex(ValueError, "desconocido"):
            Regresion(dataframe=self.data).entrenar(algoritmo="not-a-model")

    def test_external_split_integrity_and_reserved_validation(self):
        split = partition(self.data, range(24), range(24, 32), range(32, 40))
        result = Regresion(dataframe=self.data).RLM(particion=split)
        self.assertEqual(result.metadatos["train_indices"], split.train.index.tolist())
        self.assertEqual(result.metadatos["test_indices"], split.test.index.tolist())
        self.assertEqual(
            result.metadatos["validacion_indices"], split.validacion.index.tolist()
        )
        self.assertEqual(result.metadatos["semilla_particion"], 19)
        self.assertTrue(result.metadatos["particion_externa"])
        self.assertTrue(result.y_train.index.intersection(split.validacion.index).empty)
        stale = split.train.copy()
        stale.iloc[0, 0] += 1
        foreign = split.test.copy()
        foreign.index = [f"foreign-{i}" for i in range(len(foreign))]
        for bad in (
            replace(split, test=split.train),
            replace(split, train=pd.concat([split.train, split.train.iloc[:1]])),
            replace(split, train=split.train.iloc[1:]),
            replace(split, train=stale),
            replace(split, test=foreign),
            replace(split, train=split.train.drop(columns="a")),
            replace(split, validacion=split.validacion.iloc[:0]),
        ):
            with self.subTest(split=bad.huella), self.assertRaises(ValueError):
                Regresion(dataframe=self.data).RLM(particion=bad)

    def test_preprocessing_learns_no_holdout_values_or_categories(self):
        data = pd.DataFrame(
            {
                "a": [0.0, 1.0, np.nan, 3.0, 1000.0, 2000.0, 3000.0, 4000.0],
                "category": ["x", "y", "x", "y", "TEST", "TEST", "VAL", "VAL"],
                "constant": [1, 1, 1, 1, 2, 3, 4, 5],
                "target": np.arange(8) * 2.5,
            }
        )
        split = partition(data, range(4), range(4, 6), range(6, 8))
        model = Regresion(dataframe=data)
        result = model.RLM(particion=split, incluir_categoricas=True)
        pre = result.datos_preparados.transformador
        numeric = pre.transformador_.named_transformers_["numericas"]
        self.assertEqual(numeric.named_steps["imputar"].statistics_[0], 1)
        self.assertAlmostEqual(numeric.named_steps["escalar"].mean_[0], 1.25)
        self.assertEqual(pre.columnas_descartadas_, ("constant",))
        self.assertIsNone(pre.selector_)
        self.assertNotIn("TEST", " ".join(pre.get_feature_names_out()))
        self.assertNotIn("VAL", " ".join(pre.get_feature_names_out()))
        self.assertTrue(np.isfinite(model.predecir(data.iloc[4:])).all())
        with self.assertRaisesRegex(ValueError, "Faltan variables"):
            model.predecir(data.drop(columns="constant"))
        with self.assertRaisesRegex(ValueError, "faltantes"):
            Regresion(dataframe=data, features=["a"]).RLS(
                particion=split, imputar=False
            )
        with self.assertRaisesRegex(TypeError, "categóricas"):
            Regresion(dataframe=data).RLM(particion=split)

    def test_reproducibility_and_test_mutations_do_not_change_fit(self):
        first = Regresion(dataframe=self.data).RLM()
        second = Regresion(dataframe=self.data).RLM()
        assert_series_equal(first.y_pred, second.y_pred)
        self.assertEqual(first.metadatos, second.metadatos)
        changed = self.data.copy()
        test = first.y_true.index
        changed.loc[test, ["a", "b", "target"]] *= 100
        third = Regresion(dataframe=changed).RLM()
        np.testing.assert_allclose(first.modelo.coef_, third.modelo.coef_)
        assert_series_equal(first.y_pred_train, third.y_pred_train)
        self.assertNotEqual(
            first.metadatos["huella_datos"], third.metadatos["huella_datos"]
        )

    def test_small_splits_and_constant_train_fail_and_preserve_last_success(self):
        model = Regresion(dataframe=self.data)
        previous = model.RLM()
        with self.assertRaisesRegex(ValueError, "dos filas"):
            model.RLM(test_size=0.01)
        with self.assertRaises(TypeError):
            model.RLM(unknown_estimator_parameter=True)
        self.assertIs(model.modelo, previous.modelo)
        self.assertIs(model.evaluar(), previous)
        assert_series_equal(model.predecir(), previous.y_pred)
        data = self.data.copy()
        data.loc[data.index[:30], "target"] = 1
        with self.assertRaisesRegex(ValueError, "constante"):
            Regresion(dataframe=data).RLM(
                particion=partition(data, range(30), range(30, 40))
            )
        with self.assertRaisesRegex(ValueError, "dos filas"):
            Regresion(dataframe=self.data).RLM(
                particion=partition(self.data, range(1), range(1, 40))
            )


if __name__ == "__main__":
    unittest.main()
