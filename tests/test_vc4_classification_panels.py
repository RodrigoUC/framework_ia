"""Regression coverage for independent classifier panels and real comparisons."""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


class ClassificationPanelTests(unittest.TestCase):
    def test_random_forest_and_naive_bayes_have_independent_routes(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        self.assertFalse(app.exception)
        app.sidebar.button(key="nav_configuracion").click().run()
        app.radio(key="sup_tarea").set_value("clasificacion").run()
        app.sidebar.button(key="nav_clasificacion").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Random Forest" in item.value for item in app.markdown))
        self.assertFalse(any(item.value == "### Random Forest" for item in app.markdown))

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
        resultado_rf = app.session_state["resultado_lab2_modelo_RF"]
        resultado_nr = app.session_state["resultado_lab2_modelo_NR"]
        with (
            patch(
                "framework_ia.ui.lab2.Clasificacion.entrenar",
                side_effect=AssertionError("Results navigation must not train"),
            ) as entrenar,
            patch(
                "framework_ia.ui.lab2.Clasificacion.experimentar",
                side_effect=AssertionError("Results navigation must not compare"),
            ) as experimentar,
        ):
            app.sidebar.button(key="nav_lab2_resultados").click().run()
            entrenar.assert_not_called()
            experimentar.assert_not_called()
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Resultados de clasificación" in item.value for item in app.markdown)
        )
        self.assertEqual(
            sum('<h1 class="atlas-title">Resultados de clasificación</h1>' in item.value for item in app.markdown), 1
        )
        self.assertFalse(any(item.value == "### Resultados de clasificación" for item in app.markdown))
        self.assertFalse(any("Compare ejecuciones guardadas" in item.value for item in app.caption))
        self.assertFalse(app.metric)
        self.assertFalse(app.get("plotly_chart"))
        self.assertTrue(
            any(
                "Modelos entrenados individualmente" in item.value
                for item in app.markdown
            )
        )
        self.assertTrue(
            any("no con test" in item.value for item in app.caption)
        )
        for descripcion in (
            "Métricas de prueba y parámetros de los modelos entrenados individualmente",
        ):
            self.assertTrue(
                any(descripcion in item.value for item in app.caption), descripcion
            )
        tabla = next(
            item.value for item in app.dataframe if "Algoritmo" in item.value.columns
        )
        self.assertEqual(
            set(tabla["Algoritmo"]), {"Random Forest · gini", "Naive Bayes"}
        )
        for label in resultado_rf.labels:
            self.assertIn(f"Recall actual {label} (%)", tabla.columns)
        for nombre, resultado in (
            ("Random Forest · gini", resultado_rf),
            ("Naive Bayes", resultado_nr),
        ):
            fila = tabla.set_index("Algoritmo").loc[nombre]
            self.assertAlmostEqual(
                fila["Accuracy general (%)"], 100 * resultado.metricas["accuracy"]
            )
            self.assertAlmostEqual(
                fila["F1 macro prueba"], resultado.metricas["f1"]["macro"]
            )
            self.assertTrue(fila["Parámetros efectivos"])
        self.assertFalse(
            any("Comparación de modelos" in item.value for item in app.markdown)
        )
        app.sidebar.button(key="nav_clasificacion_nb").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertTrue(
            any("Valores de la matriz de confusión" in item.value for item in app.caption)
        )
        self.assertFalse(
            any("Modelos entrenados individualmente" in item.value for item in app.markdown)
        )

    def test_comparison_is_empty_until_a_model_is_trained(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        self.assertTrue(app.sidebar.button(key="nav_lab2_resultados").disabled)
        app.session_state["vista_activa"] = "lab2_resultados"
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Todavía no hay ejecuciones individuales" in item.value for item in app.info)
        )

    def test_legacy_comparison_route_migrates_to_read_only_results(self):
        app = AppTest.from_file(ENTRYPOINT, default_timeout=30).run()
        app.session_state["vista_activa"] = "comparacion"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["vista_activa"], "lab2_resultados")


if __name__ == "__main__":
    unittest.main()
