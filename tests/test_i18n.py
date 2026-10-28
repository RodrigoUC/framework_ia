"""Pruebas unitarias para el sistema de internacionalización (i18n)."""

import unittest
from framework_ia.ui.i18n import (
    cambiar_idioma,
    obtener_idioma,
    render_selector_idioma,
    t,
    IDIOMAS,
    TEXTOS,
)
from ai_framework.ui.views.navigation import _titulo_vista, _ruta_vista
import streamlit as st


class TestI18n(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()

    def test_idioma_por_defecto_es_espanol(self):
        self.assertEqual(obtener_idioma(), "es")
        self.assertEqual(t("nav_datos"), "Datos y preparación")
        self.assertEqual(t("nav_clasificacion"), "Clasificación")
        self.assertEqual(t("nav_regresion"), "Regresión")
        self.assertEqual(t("idioma_label"), "Idioma")
        self.assertEqual(t("prep_dataset"), "Preparación del dataset")
        self.assertEqual(t("prep_tipos_columna"), "Tipos de columna")
        self.assertEqual(t("configurar_tarea"), "Configurar tarea")
        self.assertEqual(_titulo_vista("datos"), "Datos y preparación")
        self.assertEqual(_ruta_vista("datos", "Datos y preparación"), "Datos / Preparación")

    def test_cambiar_idioma_a_ingles(self):
        cambiar_idioma("en")
        self.assertEqual(obtener_idioma(), "en")
        self.assertEqual(t("nav_datos"), "Data & Preparation")
        self.assertEqual(t("nav_clasificacion"), "Classification")
        self.assertEqual(t("nav_regresion"), "Regression")
        self.assertEqual(t("actualizar_graficos"), "Refresh Charts")
        self.assertEqual(t("res_modelos_ind"), "Individual Models")
        self.assertEqual(t("res_eval_comp"), "Comparative Evaluation")
        self.assertEqual(t("idioma_label"), "Language")
        self.assertEqual(t("prep_dataset"), "Dataset preparation")
        self.assertEqual(t("prep_tipos_columna"), "Column types")
        self.assertEqual(t("configurar_tarea"), "Configure task")
        self.assertEqual(t("dataset_activo"), "Active dataset")
        self.assertEqual(_titulo_vista("datos"), "Data & Preparation")
        self.assertEqual(_ruta_vista("datos", "Data & Preparation"), "Data / Preparation")

    def test_fallback_clave_inexistente(self):
        cambiar_idioma("en")
        self.assertEqual(t("clave_desconocida_xyz", fallback="Default Val"), "Default Val")
        self.assertEqual(t("clave_desconocida_xyz"), "clave_desconocida_xyz")

    def test_catalogo_idiomas_soportados(self):
        self.assertIn("es", IDIOMAS)
        self.assertIn("en", IDIOMAS)
        self.assertEqual(set(TEXTOS.keys()), {"es", "en"})


if __name__ == "__main__":
    unittest.main()
