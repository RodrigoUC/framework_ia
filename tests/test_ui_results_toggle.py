"""Pruebas unitarias para la conmutación entre modelos individuales y comparativa mediante botones."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pandas as pd
import streamlit as st

from ai_framework.ui.views.results_view import render_layout_resultados

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


class ResultsToggleTests(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()

    def test_solo_muestra_subvista_activa_clasificacion(self):
        ind_llamada = False
        comp_llamada = False

        def render_ind():
            nonlocal ind_llamada
            ind_llamada = True

        def render_comp():
            nonlocal comp_llamada
            comp_llamada = True

        with (
            patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
            patch.object(st, "button", return_value=False),
            patch.object(st, "markdown"),
        ):
            # Por defecto con ambos presentes debe mostrar individuales
            render_layout_resultados(
                hay_individuales=True,
                hay_comparativa=True,
                render_individuales_fn=render_ind,
                render_comparativa_fn=render_comp,
                clave_contexto="clasificacion",
            )
            self.assertTrue(ind_llamada)
            self.assertFalse(comp_llamada)

            # Cambiar a comparativa
            st.session_state["res_subvista_clasificacion"] = "comparativa"
            ind_llamada = False
            comp_llamada = False

            render_layout_resultados(
                hay_individuales=True,
                hay_comparativa=True,
                render_individuales_fn=render_ind,
                render_comparativa_fn=render_comp,
                clave_contexto="clasificacion",
            )
            self.assertFalse(ind_llamada)
            self.assertTrue(comp_llamada)

    def test_solo_muestra_subvista_activa_regresion(self):
        ind_llamada = False
        comp_llamada = False

        def render_ind():
            nonlocal ind_llamada
            ind_llamada = True

        def render_comp():
            nonlocal comp_llamada
            comp_llamada = True

        with (
            patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
            patch.object(st, "button", return_value=False),
            patch.object(st, "markdown"),
        ):
            # Estado inicial regresion
            render_layout_resultados(
                hay_individuales=True,
                hay_comparativa=True,
                render_individuales_fn=render_ind,
                render_comparativa_fn=render_comp,
                clave_contexto="regresion",
            )
            self.assertTrue(ind_llamada)
            self.assertFalse(comp_llamada)

            # Cambiar a comparativa
            st.session_state["res_subvista_regresion"] = "comparativa"
            ind_llamada = False
            comp_llamada = False

            render_layout_resultados(
                hay_individuales=True,
                hay_comparativa=True,
                render_individuales_fn=render_ind,
                render_comparativa_fn=render_comp,
                clave_contexto="regresion",
            )
            self.assertFalse(ind_llamada)
            self.assertTrue(comp_llamada)


if __name__ == "__main__":
    unittest.main()
