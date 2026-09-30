"""Regression coverage for independent classifier panels and real comparisons."""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from framework_ia.modelos.supervisado import Clasificacion


ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


class ClassificationPanelTests(unittest.TestCase):
    def test_training_models_separately_keeps_results_and_comparison_never_trains(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        self.assertFalse(app.exception)
        app.sidebar.button(key="nav_clasificacion").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Random Forest" in item.value for item in app.markdown))
        self.assertTrue(any("Naive Bayes" in item.value for item in app.markdown))

        rf_calls = []
        nr_calls = []
        original_rf, original_nr = Clasificacion.RF, Clasificacion.NR

        def tracked_rf(model, *args, **kwargs):
            rf_calls.append(kwargs.copy())
            return original_rf(model, *args, **kwargs)

        def tracked_nr(model, *args, **kwargs):
            nr_calls.append(kwargs.copy())
            return original_nr(model, *args, **kwargs)

        with patch.object(Clasificacion, "RF", tracked_rf), patch.object(Clasificacion, "NR", tracked_nr):
            app.button(key="entrenar_clasificacion_rf").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(rf_calls), 1)
            self.assertEqual(nr_calls, [])
            app.button(key="entrenar_clasificacion_nr").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(rf_calls), 1)
            self.assertEqual(len(nr_calls), 1)

            trained = app.session_state["resultado_modelos_clasificacion"]
            context_signature, entries = next(iter(trained.items()))
            self.assertEqual(set(entries), {"RF", "NR"})
            self.assertIn(app.session_state["dataset_identidad"], context_signature)
            app.sidebar.button(key="nav_comparacion").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(rf_calls), 1)
            self.assertEqual(len(nr_calls), 1)

        comparison = app.dataframe[-1].value
        self.assertEqual(set(comparison["algoritmo"]), {"Random Forest", "Naive Bayes"})
        self.assertTrue(comparison["hiperparámetros efectivos"].str.contains("n_estimators").any())
        self.assertTrue(comparison["hiperparámetros efectivos"].str.contains("var_smoothing").any())
        self.assertTrue(comparison["accuracy"].between(0, 1).all())
        self.assertTrue(comparison["precision"].between(0, 1).all())

        app.sidebar.button(key="nav_clasificacion").click().run()
        self.assertFalse(app.exception)
        current_estimators = app.session_state["clasif_rf_n"]
        app.slider(key="clasif_rf_n").set_value(current_estimators - 10).run()
        self.assertFalse(app.exception)
        app.sidebar.button(key="nav_comparacion").click().run()
        self.assertFalse(app.exception)
        filtered_comparison = app.dataframe[-1].value
        self.assertEqual(filtered_comparison["algoritmo"].tolist(), ["Naive Bayes"])

    def test_comparison_is_empty_until_a_model_is_trained(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        app.sidebar.button(key="nav_comparacion").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Entrene al menos un modelo" in item.value for item in app.info))
        self.assertFalse(app.dataframe)


if __name__ == "__main__":
    unittest.main()
