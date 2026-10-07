"""LAB02 software checks; generated fixtures are not academic experiment results."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_ia.datos.particion import Particionador
from framework_ia.modelos.supervisado import Clasificacion, Supervisado
from framework_ia.modelos.supervisado.clasificacion import configuraciones_lab2
from framework_ia.modelos.supervisado.preprocesamiento import PreprocesadorSupervisado
from framework_ia.resultados import ResultadoParticion
from framework_ia.utils import ConfiguracionParticion


def synthetic(n=120, classes=2):
    X, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=73,
    )
    data = pd.DataFrame(X, columns=list("abcde"))
    data["target"] = y
    data.index = pd.Index([f"row-{n}" for n in range(len(data))])
    return data


def partition(data, train, test, validation=None):
    validation = [] if validation is None else validation
    return ResultadoParticion(
        train=data.iloc[train].copy(),
        test=data.iloc[test].copy(),
        validacion=data.iloc[validation].copy() if len(validation) else None,
        porcentaje_train=len(train) / len(data),
        porcentaje_test=len(test) / len(data),
        porcentaje_validacion=len(validation) / len(data),
        columna_estratificacion="target",
        distribucion_train=None,
        distribucion_test=None,
        distribucion_validacion=None,
        semilla=13,
    )


class ClasificacionLab2Tests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic()

    def test_algorithms_and_pythonic_aliases_preserve_numeric_labels(self):
        self.assertTrue(issubclass(Clasificacion, Supervisado))
        for method in (
            "knn",
            "arbol_decision",
            "random_forest",
            "xgboost",
            "adaboost",
            "naive_bayes",
        ):
            with self.subTest(method=method):
                c = Clasificacion(dataframe=self.data, target="target")
                kwargs = (
                    {"n_estimators": 8}
                    if method in ("random_forest", "xgboost", "adaboost")
                    else {}
                )
                result = getattr(c, method)(**kwargs)
                self.assertEqual(result.labels, (0, 1))
                self.assertEqual(
                    result.y_pred.index.tolist(), result.metadatos["test_indices"]
                )
                self.assertEqual(result.muestra_test, 30)
                self.assertTrue(
                    result.metadatos["preprocesamiento"]["ajustado_solo_train"]
                )
                assert_series_equal(c.predecir(), result.y_pred)
                self.assertIn(
                    "random_state",
                    result.parametros
                    if method not in ("knn", "naive_bayes")
                    else {"random_state": None},
                )
        for alias in (
            "KNN",
            "DT",
            "RF",
            "XGBoost",
            "XGB",
            "AdaBoost",
            "ADA",
            "NR",
            "dt",
            "rf",
            "xgb",
            "ada",
            "nr",
        ):
            self.assertTrue(callable(getattr(Clasificacion, alias)))

    def test_xgboost_decodes_string_and_nonconsecutive_multiclass_labels(self):
        for mapping in ({0: "alta", 1: "baja", 2: "media"}, {0: 2, 1: 7, 2: 90}):
            with self.subTest(mapping=mapping):
                data = synthetic(classes=3)
                data["target"] = data.target.map(mapping)
                c = Clasificacion(dataframe=data, target="target")
                result = c.xgboost(n_estimators=8, max_depth=2)
                self.assertEqual(set(result.labels), set(mapping.values()))
                pred = c.predecir(data.drop(columns="target").iloc[:6])
                self.assertTrue(set(pred).issubset(set(mapping.values())))
                self.assertEqual(pred.index.tolist(), data.index[:6].tolist())
                raw = result.modelo.predict(
                    c._transformar(data.drop(columns="target").iloc[:6])
                )
                np.testing.assert_array_equal(raw, pred.values)

    def test_preprocessing_fit_only_train_not_held_out_values_or_categories(self):
        data = pd.DataFrame(
            {
                "x": [1.0, 2.0, np.nan, 4.0, 1000.0, 2000.0],
                "category": ["a", "b", None, "a", "TEST_ONLY", "TEST_ONLY"],
                "constant_train": [5.0, 5.0, 5.0, 5.0, 10.0, 20.0],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        split = partition(data, [0, 1, 2, 3], [4, 5])
        c = Clasificacion(dataframe=data, target="target")
        result = c.DT(particion=split, incluir_categoricas=True)
        pre = c._preparados.transformador
        numeric = pre.transformador_.named_transformers_["numericas"]
        self.assertEqual(numeric.named_steps["imputar"].statistics_[0], 2.0)
        self.assertAlmostEqual(numeric.named_steps["escalar"].mean_[0], 2.25)
        categories = (
            pre.transformador_.named_transformers_["categoricas"]
            .named_steps["codificar"]
            .categories_[0]
        )
        self.assertEqual(set(categories), {"a", "b"})
        self.assertIn("constant_train", result.metadatos["columnas_descartadas"])
        self.assertNotIn(
            "TEST_ONLY", " ".join(result.metadatos["columnas_transformadas"])
        )
        transformed = c._transformar(data.iloc[4:].drop(columns="target"))
        self.assertTrue(np.isfinite(transformed.values).all())
        self.assertEqual(len(c.predecir(data.iloc[4:].drop(columns="target"))), 2)

    def test_inference_cleans_infinities_with_training_imputer(self):
        c = Clasificacion(dataframe=self.data, target="target")
        c.DT()
        new = self.data.iloc[:3].drop(columns="target").copy()
        new.iloc[0, 0] = np.inf
        self.assertFalse(c.predecir(new).isna().any())
        with self.assertRaisesRegex(ValueError, "Faltan variables"):
            c.predecir(new.drop(columns="a"))
        with self.assertRaises(TypeError):
            c.predecir(new.iloc[0])

    def test_preprocessor_rejects_missing_without_imputation_and_category_opt_out(self):
        data = pd.DataFrame({"x": [1.0, np.nan, 3.0], "s": ["a", "b", "a"]})
        with self.assertRaisesRegex(ValueError, "faltantes"):
            PreprocesadorSupervisado(imputar=False).fit(data[["x"]])
        with self.assertRaisesRegex(TypeError, "categóricas"):
            PreprocesadorSupervisado().fit(data)
        with self.assertRaisesRegex(ValueError, "variación"):
            PreprocesadorSupervisado().fit(pd.DataFrame({"x": [np.nan, np.nan]}))

    def test_preview_preserves_original_nonrange_indices_and_null_target_filter(self):
        data = self.data.copy()
        data["target"] = data.target.astype(float)
        data.loc[data.index[0], "target"] = np.nan
        c = Clasificacion(dataframe=data, target="target")
        preview = c.previsualizar_particion(random_state=19)
        result = c.RF(n_estimators=5, random_state=19)
        self.assertEqual(preview["test_indices"], result.metadatos["test_indices"])
        self.assertEqual(preview["train_indices"], result.metadatos["train_indices"])
        self.assertNotIn(
            data.index[0], preview["train_indices"] + preview["test_indices"]
        )
        self.assertEqual(result.metadatos["filas_objetivo_nulo"], 1)

    def test_validation_partition_is_reused_exactly(self):
        split = Particionador(
            ConfiguracionParticion(
                porcentaje_test=0.2,
                porcentaje_validacion=0.2,
                columna_estratificacion="target",
                semilla=10,
            )
        ).dividir(self.data)
        result = Clasificacion(dataframe=self.data, target="target").experimentar(
            algoritmos=["DT"],
            particion=split,
        )
        self.assertEqual(result.metadatos["train_indices"], split.train.index.tolist())
        self.assertEqual(result.metadatos["test_indices"], split.test.index.tolist())
        self.assertEqual(
            result.metadatos["validacion_indices"], split.validacion.index.tolist()
        )
        self.assertTrue(result.metadatos["validacion_externa"])

    def test_external_train_test_gets_validation_only_from_train(self):
        split = Particionador(
            ConfiguracionParticion(
                porcentaje_test=0.25,
                columna_estratificacion="target",
                semilla=10,
            )
        ).dividir(self.data)
        result = Clasificacion(dataframe=self.data, target="target").experimentar(
            algoritmos=["DT"], particion=split
        )
        meta = result.metadatos
        self.assertEqual(meta["test_indices"], split.test.index.tolist())
        self.assertEqual(
            set(meta["train_indices"] + meta["validacion_indices"]),
            set(split.train.index),
        )
        self.assertFalse(set(meta["train_indices"]) & set(meta["validacion_indices"]))

    def test_external_split_rejects_overlap_duplicates_omissions_foreign_and_stale_data(
        self,
    ):
        good = partition(self.data, list(range(90)), list(range(90, 120)))
        stale = good.train.copy()
        stale.iloc[0, 0] += 1
        foreign = good.test.copy()
        foreign.index = ["foreign"] + foreign.index[1:].tolist()
        changes = [
            (
                replace(good, test=pd.concat([good.test, good.train.iloc[:1]])),
                "solapados",
            ),
            (
                replace(good, train=pd.concat([good.train, good.train.iloc[:1]])),
                "duplicados",
            ),
            (replace(good, train=good.train.iloc[1:]), "cubrir"),
            (replace(good, train=stale), "obsoleta"),
            (replace(good, test=foreign), "ajenos"),
            (replace(good, test=good.test.drop(columns="a")), "columnas"),
            (replace(good, test=good.test.iloc[:0]), "vacía"),
        ]
        for split, message in changes:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                Clasificacion(dataframe=self.data, target="target").DT(particion=split)

    def test_data_validation_rejects_continuous_single_class_and_duplicate_indices(
        self,
    ):
        for invalid in (
            self.data.assign(target=np.linspace(0.01, 0.99, len(self.data))),
            self.data.assign(target=0),
        ):
            with self.assertRaises((ValueError, TypeError)):
                Clasificacion(dataframe=invalid, target="target").DT()
        duplicated = self.data.copy()
        duplicated.index = [0] * len(duplicated)
        with self.assertRaisesRegex(ValueError, "índices únicos"):
            Clasificacion(dataframe=duplicated, target="target").DT()
        with self.assertRaisesRegex(ValueError, "target"):
            Clasificacion(
                dataframe=self.data, target="target", features=["a", "target"]
            ).DT()
        with self.assertRaisesRegex(ValueError, "No hay datos"):
            Clasificacion(dataframe=pd.DataFrame()).DT()

    def test_unseen_holdout_class_and_too_few_stratification_rows_fail_explicitly(self):
        data = pd.DataFrame({"x": range(6), "target": [0, 1, 0, 1, 2, 2]})
        split = partition(data, [0, 1, 2, 3], [4, 5])
        with self.assertRaisesRegex(ValueError, "ausentes"):
            Clasificacion(dataframe=data, target="target").DT(particion=split)
        data.loc[5, "target"] = 3
        with self.assertRaisesRegex(ValueError, "partición"):
            Clasificacion(dataframe=data, target="target").DT()

    def test_default_catalog_is_fresh_and_covers_five_standard_plus_variants(self):
        first = configuraciones_lab2()
        self.assertEqual(set(first), {"KNN", "DT", "RF", "XGBoost", "AdaBoost"})
        self.assertTrue(
            all(
                len(configs) == 3 and configs[0]["nombre"] == "estandar"
                for configs in first.values()
            )
        )
        first["KNN"][0]["parametros"]["n_neighbors"] = 100
        self.assertEqual(configuraciones_lab2()["KNN"][0]["parametros"], {})

    def test_benchmark_all_five_reproducible_and_exports_effective_parameters(self):
        first = Clasificacion(dataframe=self.data, target="target").experimentar(
            random_state=23
        )
        second = Clasificacion(dataframe=self.data, target="target").experimentar(
            random_state=23
        )
        assert_frame_equal(first.tabla, second.tabla)
        self.assertEqual(len(first.tabla), 15)
        self.assertEqual(first.metadatos["candidatos_fallidos"], 0)
        self.assertEqual(
            first.metadatos["test_indices"], second.metadatos["test_indices"]
        )
        self.assertEqual(first.mejor_algoritmo, second.mejor_algoritmo)
        self.assertEqual(first.tabla.seleccionado.sum(), 5)
        self.assertFalse(any("test" in column for column in first.tabla))
        standard_knn = first.tabla[
            (first.tabla.algoritmo == "KNN") & (first.tabla.configuracion == "estandar")
        ].iloc[0]
        self.assertEqual(standard_knn.parametros["n_neighbors"], 5)
        for algorithm, result in first.mejores.items():
            assert_series_equal(result.y_pred, second.mejores[algorithm].y_pred)
            self.assertEqual(result.metadatos["conjunto_evaluacion"], "test")
            self.assertIn("metricas_validacion", result.metadatos)

    def test_test_mutation_cannot_change_validation_scores_selection_or_training(self):
        split = Particionador(
            ConfiguracionParticion(
                porcentaje_test=0.2,
                porcentaje_validacion=0.2,
                columna_estratificacion="target",
                semilla=9,
            )
        ).dividir(self.data)
        first = Clasificacion(dataframe=self.data, target="target").experimentar(
            algoritmos=["KNN", "DT"], particion=split
        )
        changed = self.data.copy()
        changed.loc[split.test.index, "target"] = (
            1 - changed.loc[split.test.index, "target"]
        )
        changed.loc[split.test.index, "a"] *= 10000
        split2 = replace(split, test=changed.loc[split.test.index].copy())
        second = Clasificacion(dataframe=changed, target="target").experimentar(
            algoritmos=["KNN", "DT"], particion=split2
        )
        assert_frame_equal(first.tabla, second.tabla)
        self.assertEqual(first.mejor_algoritmo, second.mejor_algoritmo)
        self.assertNotEqual(
            first.metadatos["huella_datos"], second.metadatos["huella_datos"]
        )

    def test_all_validation_candidates_are_evaluated_before_any_test(self):
        c = Clasificacion(dataframe=self.data, target="target")
        calls = []
        original = c._resultado

        def tracking(*args, **kwargs):
            metadata = args[-1]
            calls.append(metadata["conjunto_evaluacion"])
            return original(*args, **kwargs)

        with patch.object(c, "_resultado", side_effect=tracking):
            c.experimentar(algoritmos=["KNN", "DT"])
        self.assertEqual(calls, ["validacion"] * 6 + ["test"] * 2)

    def test_candidate_failures_are_isolated_and_clear_old_success(self):
        c = Clasificacion(dataframe=self.data, target="target")
        c.DT()
        result = c.experimentar(
            algoritmos=["KNN", "DT"],
            configuraciones={
                "KNN": [{"nombre": "invalid", "parametros": {"n_neighbors": 10000}}],
                "DT": [{"nombre": "valid", "parametros": {"max_depth": 3}}],
            },
        )
        self.assertEqual(result.metadatos["candidatos_fallidos"], 1)
        self.assertEqual(set(result.mejores), {"DT"})
        self.assertIn("ValueError", result.tabla.loc[0, "error"])
        result = c.experimentar(
            algoritmos=["KNN"],
            configuraciones={
                "KNN": [{"nombre": "invalid", "parametros": {"n_neighbors": 10000}}],
            },
        )
        self.assertEqual(result.mejores, {})
        self.assertIsNone(result.mejor_algoritmo)
        with self.assertRaises(RuntimeError):
            c.evaluar()
        c.DT()
        with self.assertRaises(ValueError):
            c.experimentar(metrica="test_accuracy")
        self.assertIsNone(c.modelo)
        self.assertIsNone(c.experimento)

    def test_configuration_errors_are_explicit(self):
        for kwargs in (
            {"algoritmos": []},
            {"algoritmos": ["invalid"]},
            {"metrica": "test_accuracy"},
            {"validation_size": 0},
            {"configuraciones": {"DT": []}},
            {
                "configuraciones": {
                    "DT": [{"nombre": "x", "parametros": {}, "unknown": 1}]
                }
            },
            {
                "configuraciones": {
                    "XGB": [{"nombre": "x", "parametros": {"eval_set": []}}]
                }
            },
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises((ValueError, TypeError)),
            ):
                Clasificacion(dataframe=self.data, target="target").experimentar(
                    **kwargs
                )


if __name__ == "__main__":
    unittest.main()
