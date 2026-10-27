"""Native regression estimators and deterministic Lab03 configurations."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from framework_ia import Regresion
from framework_ia.utils.supervisado.algoritmos_regresion import (
    crear_modelo,
    normalizar_algoritmo,
)
from framework_ia.datos.preprocesamiento import PreprocesadorSupervisado
from framework_ia.utils import (
    ALGORITMOS_LAB3,
    CONFIGURACIONES_REGRESION,
    NOMBRES_REGRESORES,
    configuraciones_experimento_regresion,
)
from framework_ia.utils.configuracion_regresion import ALPHAS_LAB3


class RegressionEstimatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(2, 80))
        self.data = pd.DataFrame(
            {"a": a, "b": b, "target": a * 2 + b * 3 + rng.normal(size=80)}
        )

    def test_nine_native_estimators_pythonic_and_compatibility_methods(self):
        methods = (
            ("RLS", "lineal_simple", LinearRegression),
            ("RLM", "lineal_multiple", LinearRegression),
            ("Lasso", "lasso", Lasso),
            ("LassoCV", "lasso_cv", LassoCV),
            ("Ridge", "ridge", Ridge),
            ("RidgeCV", "ridge_cv", RidgeCV),
            ("SVR", "svr", SVR),
            ("DT", "arbol_decision", DecisionTreeRegressor),
            ("RF", "random_forest", RandomForestRegressor),
        )
        for canonical, pythonic, estimator in methods:
            results = []
            for method in (canonical, pythonic):
                with self.subTest(method=method):
                    model = Regresion(
                        dataframe=self.data,
                        features=["a"] if canonical == "RLS" else None,
                    )
                    result = getattr(model, method)()
                    self.assertIs(type(result.modelo), estimator)
                    self.assertEqual(result.algoritmo, canonical)
                    self.assertEqual(len(result.y_pred), 20)
                    self.assertTrue(all(np.isfinite(list(result.metricas.values()))))
                    self.assertIsNone(result.datos_preparados.transformador.selector_)
                    json.dumps(result.parametros, allow_nan=False)
                    json.dumps(result.metadatos, allow_nan=False)
                    assert_series_equal(model.predecir(), result.y_pred)
                    results.append(result)
            assert_series_equal(results[0].y_pred, results[1].y_pred)

    def test_catalog_has_exact_fourteen_configs_and_fresh_containers(self):
        catalog = configuraciones_experimento_regresion()
        self.assertEqual(tuple(catalog), ALGORITMOS_LAB3)
        self.assertEqual(tuple(CONFIGURACIONES_REGRESION), ALGORITMOS_LAB3)
        self.assertEqual(tuple(NOMBRES_REGRESORES), ALGORITMOS_LAB3)
        self.assertEqual(sum(map(len, catalog.values())), 14)
        for algorithm, candidates in catalog.items():
            self.assertEqual(
                len({item["nombre"] for item in candidates}), len(candidates)
            )
            for candidate in candidates:
                with self.subTest(algorithm=algorithm, name=candidate["nombre"]):
                    result = Regresion(
                        dataframe=self.data,
                        features=["a"] if algorithm == "RLS" else None,
                    ).entrenar(algoritmo=algorithm, **candidate["parametros"])
                    for key, value in candidate["parametros"].items():
                        self.assertEqual(result.parametros[key], value)
        self.assertEqual(
            [c["parametros"]["alpha"] for c in catalog["Lasso"]], [0.1, 1.0]
        )
        self.assertEqual(
            [c["parametros"]["alpha"] for c in catalog["Ridge"]], [0.1, 1.0]
        )
        self.assertEqual([c["parametros"]["C"] for c in catalog["SVR"]], [1.0, 10.0])
        self.assertEqual(
            [c["parametros"]["max_depth"] for c in catalog["DT"]], [None, 5]
        )
        self.assertEqual(
            [c["parametros"]["max_depth"] for c in catalog["RF"]], [None, 5]
        )
        self.assertTrue(
            all(c["parametros"]["n_estimators"] == 100 for c in catalog["RF"])
        )
        catalog["Lasso"][0]["parametros"]["alpha"] = 99
        catalog["RF"].clear()
        fresh = configuraciones_experimento_regresion()
        self.assertEqual(fresh["Lasso"][0]["parametros"]["alpha"], 0.1)
        self.assertEqual(len(fresh["RF"]), 2)
        json.dumps(fresh)

    def test_ui_parameter_defaults_match_native_estimators(self):
        for algorithm, configuration in CONFIGURACIONES_REGRESION.items():
            parameters = crear_modelo(algorithm, 42, {}).get_params()
            for item in configuration.parametros:
                with self.subTest(algorithm=algorithm, parameter=item.nombre):
                    default = item.opciones[0] if item.tipo == "select" else item.valor
                    if item.tipo == "depth" and default == 0:
                        default = None
                    self.assertEqual(default, parameters[item.nombre])

    def test_cv_native_alpha_selection_and_serializable_splitter(self):
        for algorithm, constructor in (("LassoCV", LassoCV), ("RidgeCV", RidgeCV)):
            with self.subTest(algorithm=algorithm):
                fit_calls = []
                original = PreprocesadorSupervisado.fit

                def track(preprocessor, X, y=None, calls=fit_calls, fit=original):
                    calls.append(X.index.tolist())
                    return fit(preprocessor, X, y)

                with patch.object(PreprocesadorSupervisado, "fit", track):
                    result = Regresion(dataframe=self.data).entrenar(
                        algoritmo=algorithm, random_state=17
                    )
                self.assertEqual(fit_calls, [result.metadatos["train_indices"]])
                self.assertIsInstance(result.modelo.cv, KFold)
                cv_description = {
                    "tipo": "KFold",
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 17,
                }
                self.assertEqual(result.parametros["cv"], cv_description)
                self.assertEqual(result.parametros["alphas"], ALPHAS_LAB3)
                meta = result.metadatos["cv_interna"]
                self.assertFalse(meta["preprocesamiento_por_pliegue"])
                self.assertEqual(meta["uso"], "seleccion_alpha_solo_train")
                self.assertEqual(
                    result.metadatos["alpha_elegido"], float(result.modelo.alpha_)
                )
                self.assertIn(result.modelo.alpha_, ALPHAS_LAB3)
                kwargs = {
                    "alphas": ALPHAS_LAB3,
                    "cv": KFold(5, shuffle=True, random_state=17),
                }
                if algorithm == "RidgeCV":
                    kwargs["scoring"] = "neg_mean_squared_error"
                    self.assertEqual(result.modelo.scoring, "neg_mean_squared_error")
                else:
                    kwargs.update(random_state=17, n_jobs=1)
                native = constructor(**kwargs).fit(
                    result.datos_preparados.matriz, result.y_train
                )
                self.assertEqual(native.alpha_, result.modelo.alpha_)
                np.testing.assert_allclose(native.coef_, result.modelo.coef_)

    def test_seeds_custom_parameters_and_single_feature_non_linear_models(self):
        for algorithm in ("Lasso", "LassoCV", "Ridge", "DT", "RF"):
            with self.subTest(algorithm=algorithm):
                result = Regresion(dataframe=self.data, features=["a"]).entrenar(
                    algoritmo=algorithm, random_state=9
                )
                self.assertEqual(result.parametros["random_state"], 9)
        result = Regresion(dataframe=self.data).ridge(alpha=2.5, fit_intercept=False)
        self.assertEqual(result.parametros["alpha"], 2.5)
        self.assertFalse(result.parametros["fit_intercept"])
        result = Regresion(dataframe=self.data).svr(C=10, epsilon=0.2, kernel="linear")
        self.assertEqual(
            (result.parametros["C"], result.parametros["epsilon"]), (10, 0.2)
        )
        result = Regresion(dataframe=self.data).lasso_cv(alphas=np.array([0.1, 1.0]))
        self.assertEqual(result.parametros["alphas"], [0.1, 1.0])
        json.dumps(result.parametros)

    def test_invalid_configuration_and_small_cv_preserve_previous_success(self):
        model = Regresion(dataframe=self.data)
        previous = model.RLM()
        for algorithm, parameters in (
            ("Lasso", {"alpha": -1}),
            ("Ridge", {"alpha": -1}),
            ("LassoCV", {"cv": 3}),
            ("RidgeCV", {"scoring": "r2"}),
            ("RidgeCV", {"alphas": []}),
            ("SVR", {"C": 0}),
            ("DT", {"max_depth": -1}),
            ("RF", {"n_estimators": 0}),
            ("RF", {"unknown_parameter": True}),
        ):
            with self.subTest(algorithm=algorithm, parameters=parameters):
                with self.assertRaises((ValueError, TypeError)):
                    model.entrenar(algoritmo=algorithm, **parameters)
                self.assertIs(model.evaluar(), previous)
        for algorithm in ("LassoCV", "RidgeCV"):
            with self.subTest(algorithm=algorithm), self.assertRaises(ValueError):
                Regresion(dataframe=self.data.iloc[:6]).entrenar(algoritmo=algorithm)
        self.assertEqual(normalizar_algoritmo(" ridge_cv "), "RidgeCV")
        with self.assertRaises(ValueError):
            normalizar_algoritmo(None)

    def test_holdout_changes_cannot_affect_cv_alpha_or_forest_training(self):
        for algorithm in ("LassoCV", "RidgeCV", "RF"):
            with self.subTest(algorithm=algorithm):
                first = Regresion(dataframe=self.data).entrenar(algoritmo=algorithm)
                changed = self.data.copy()
                changed.loc[first.y_true.index, ["a", "b", "target"]] *= 1000
                second = Regresion(dataframe=changed).entrenar(algoritmo=algorithm)
                assert_series_equal(first.y_pred_train, second.y_pred_train)
                if algorithm.endswith("CV"):
                    self.assertEqual(first.modelo.alpha_, second.modelo.alpha_)
                    np.testing.assert_allclose(first.modelo.coef_, second.modelo.coef_)


if __name__ == "__main__":
    unittest.main()
