"""Regression coverage for independent classifier panels and real comparisons."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


class ClassificationPanelTests(unittest.TestCase):
    def test_random_forest_and_naive_bayes_have_independent_routes(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        self.assertFalse(app.exception)
        app.sidebar.button(key="nav_clasificacion").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Random Forest" in item.value for item in app.markdown))

        app.button(key="lab2_entrenar_RF").click().run()
        self.assertFalse(app.exception)
        self.assertIn("resultado_lab2_modelo_RF", app.session_state)
        app.sidebar.button(key="nav_clasificacion_nb").click().run()
        app.button(key="lab2_entrenar_NR").click().run()
        self.assertFalse(app.exception)

        self.assertIn("resultado_lab2_modelo_NR", app.session_state)
        self.assertIsNot(
            app.session_state["resultado_lab2_modelo_RF"],
            app.session_state["resultado_lab2_modelo_NR"],
        )
        app.sidebar.button(key="nav_comparacion").click().run()
        self.assertFalse(app.exception)

    def test_comparison_is_empty_until_a_model_is_trained(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        app.sidebar.button(key="nav_comparacion").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Entrene al menos un modelo" in item.value for item in app.info)
        )


if __name__ == "__main__":
    unittest.main()
