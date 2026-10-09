"""Regression tests for the source, navigation, and analysis-configuration flow."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


class VisualContrastFlowTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(ENTRYPOINT, default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def test_source_is_collapsed_and_changes_apply_only_on_submit(self):
        source = next(item for item in self.app.sidebar.expander if item.label == "Fuente de datos")
        self.assertFalse(source.proto.expanded)
        active_identity = self.app.session_state["dataset_identidad"]

        self.app.sidebar.text_input(key="fuente_separador").set_value(";").run()
        self.assertEqual(active_identity, self.app.session_state["dataset_identidad"])
        self.assertFalse(self.app.exception)

        # AppTest batches form edits with submit, just like the browser.
        self.app.sidebar.text_input(key="fuente_separador").set_value(";")
        self.app.button(key="FormSubmitter:form_fuente_datos-Aplicar fuente").click().run()
        self.assertFalse(self.app.exception)
        self.assertNotEqual(active_identity, self.app.session_state["dataset_identidad"])
        self.assertEqual(len(self.app.session_state["dataset_preparado"].columns), 1)
        applied_identity = self.app.session_state["dataset_identidad"]
        self.app.sidebar.button(key="nav_datos").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(applied_identity, self.app.session_state["dataset_identidad"])

    def test_theme_refresh_reruns_without_clearing_applied_data_or_results(self):
        active_identity = self.app.session_state["dataset_identidad"]
        preserved_result = {"firma": ("test",), "resultado": "resultado conservado"}
        self.app.session_state["resultado_acp"] = preserved_result

        refresh = self.app.sidebar.button(key="actualizar_graficos")
        self.assertEqual(refresh.label, "Actualizar gráficos")
        self.assertIn("después de cambiar el tema", refresh.proto.help)

        refresh.click().run()

        self.assertFalse(self.app.exception)
        self.assertEqual(active_identity, self.app.session_state["dataset_identidad"])
        self.assertEqual(preserved_result, self.app.session_state["resultado_acp"])

    def test_navigation_follows_analysis_to_results_and_uses_section_label(self):
        labels = [item.label for item in self.app.sidebar.button]
        self.assertLess(labels.index("Datos y preparación"), labels.index("EDA y ACP"))
        self.assertLess(labels.index("EDA y ACP"), labels.index("K-Means"))
        self.assertLess(labels.index("K-Means"), labels.index("K-Medoids"))
        self.assertLess(labels.index("K-Medoids"), labels.index("HAC"))
        self.assertLess(labels.index("HAC"), labels.index("Random Forest y Naive Bayes"))
        self.assertLess(labels.index("Random Forest y Naive Bayes"), labels.index("Comparar modelos"))
        markdown = " ".join(item.value for item in self.app.sidebar.markdown)
        self.assertIn("Análisis y modelos", markdown)
        self.assertNotIn("Pilares del framework", markdown)

    def test_analysis_configuration_is_visible_in_main_content(self):
        self.app.sidebar.button(key="nav_eda").click().run()
        self.assertFalse(self.app.exception)
        main_markdown = " ".join(item.value for item in self.app.markdown)
        sidebar_markdown = " ".join(item.value for item in self.app.sidebar.markdown)
        self.assertIn("Configuración de agrupamiento y reducción", main_markdown)
        self.assertNotIn("Configuración de agrupamiento y reducción", sidebar_markdown)
        self.assertTrue(any(item.label == "Análisis de componentes principales (ACP)" for item in self.app.expander))
        self.assertIn("atlas-title", main_markdown)

    def test_invalid_default_csv_keeps_app_available_for_source_recovery(self):
        def run_with_broken_csv():
            # AppTest executes this function in an isolated script namespace.
            from unittest.mock import patch

            from framework_ia.ui import streamlit_app

            with patch.object(
                streamlit_app.CargadorCSV,
                "cargar_ruta",
                side_effect=ValueError("invalid CSV fixture"),
            ):
                streamlit_app.main()

        app = AppTest.from_function(run_with_broken_csv, default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Elija otra fuente" in item.value for item in app.sidebar.error)
        )
        app.sidebar.radio(key="fuente_origen").set_value("Subir CSV").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.sidebar.file_uploader(key="fuente_upload"))


if __name__ == "__main__":
    unittest.main()
