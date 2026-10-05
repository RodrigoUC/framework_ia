"""Pruebas de regresión para el contraste de gráficos Plotly por tema."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from framework_ia.datos.eda import EDA
from framework_ia.ui import streamlit_app
from framework_ia.visualizacion import (
    VisualizadorDatos,
    VisualizadorNoSupervisado,
    VisualizadorSupervisado,
)


class ContrasteTemaPlotlyTests(unittest.TestCase):
    def test_keyboard_focus_outline_has_contrast_in_both_themes(self):
        casos = (("light", "#1a3c2b", "#f7f7f5"), ("dark", "#9effbf", "#111713"))
        for theme, focus_color, background in casos:
            with self.subTest(theme=theme):
                context = SimpleNamespace(theme=SimpleNamespace(type=theme))
                with (
                    patch.object(streamlit_app.st, "context", context),
                    patch.object(streamlit_app.st, "markdown") as markdown,
                ):
                    streamlit_app._aplicar_estilos_atlas()

                stylesheet = markdown.call_args.args[0]
                self.assertIn(
                    f":root {{ --atlas-focus-color: {focus_color}; }}", stylesheet
                )
                self.assertIn("outline: 2px solid var(--atlas-focus-color)", stylesheet)
                self.assertIn("input:focus-visible", stylesheet)
                self.assertIn(
                    '[data-testid="stSidebar"] .stButton > button:focus-visible',
                    stylesheet,
                )
                self.assertIn("outline-color: #f4d35e !important", stylesheet)
                self.assertGreaterEqual(self._contraste(focus_color, background), 3)

    def test_multiselect_chip_palette_meets_wcag_in_both_themes(self):
        casos = (
            ("light", "#1a3c2b", "#f7f7f5"),
            ("dark", "#9effbf", "#111713"),
        )
        for theme, background, foreground in casos:
            with self.subTest(theme=theme):
                context = SimpleNamespace(theme=SimpleNamespace(type=theme))
                with (
                    patch.object(streamlit_app.st, "context", context),
                    patch.object(streamlit_app.st, "markdown") as markdown,
                ):
                    streamlit_app._aplicar_estilos_atlas()

                stylesheet = markdown.call_args.args[0]
                self.assertIn(
                    '[data-testid="stMultiSelect"] [data-tag]', stylesheet
                )
                self.assertIn(f"background-color: {background}", stylesheet)
                self.assertIn(f"color: {foreground}", stylesheet)
                self.assertIn("[data-tag] svg", stylesheet)
                self.assertNotIn("        [data-tag] {", stylesheet)
                self.assertGreaterEqual(
                    self._contraste(foreground, background), 4.5
                )

    def test_fondo_ejes_y_anotaciones_siguen_el_tema_activo(self):
        casos = (
            (False, "#f7f7f5", "#eeeee9", "#242523"),
            (True, "#111713", "#1a241d", "#f7f7f5"),
        )
        for oscuro, fondo, superficie, texto in casos:
            with self.subTest(oscuro=oscuro):
                figura = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
                figura.add_annotation(x=0, y=0, text="Etiqueta", showarrow=True)
                streamlit_app._aplicar_tema_figura(figura, oscuro)

                self.assertEqual(figura.layout.paper_bgcolor, fondo)
                self.assertEqual(figura.layout.plot_bgcolor, superficie)
                self.assertEqual(figura.layout.font.color, texto)
                self.assertEqual(figura.layout.xaxis.color, texto)
                self.assertEqual(figura.layout.yaxis.color, texto)
                self.assertIsNotNone(figura.layout.xaxis.gridcolor)
                self.assertIsNotNone(figura.layout.yaxis.gridcolor)
                self.assertEqual(figura.layout.annotations[0].font.color, texto)

    def test_colores_de_traza_de_bajo_contraste_se_ajustan_sin_cambiar_datos(self):
        figura = go.Figure(
            go.Scatter(
                x=[1, 2],
                y=[3, 4],
                mode="markers+lines",
                marker={"color": "#1a3c2b"},
                line={"color": "#1a3c2b"},
            )
        )
        streamlit_app._aplicar_tema_figura(figura, oscuro=True)

        self.assertEqual(list(figura.data[0].x), [1, 2])
        self.assertEqual(list(figura.data[0].y), [3, 4])
        self.assertNotEqual(figura.data[0].marker.color, "#1a3c2b")
        self.assertEqual(figura.data[0].marker.color, figura.data[0].line.color)
        self.assertGreaterEqual(
            self._contraste(
                figura.data[0].marker.color, streamlit_app.PALETA_OSCURA["superficie"]
            ),
            3,
        )
        self.assertNotEqual(figura.data[0].marker.color, "#ffffff")

    def test_colores_de_traza_conservan_matiz_en_ambos_temas(self):
        for oscuro, color, fondo in (
            (False, "#ff8c69", streamlit_app.PALETA["superficie"]),
            (True, "#1a3c2b", streamlit_app.PALETA_OSCURA["superficie"]),
        ):
            with self.subTest(oscuro=oscuro):
                ajustado = streamlit_app._ajustar_color_traza(color, oscuro)
                canales = streamlit_app._color_rgb(ajustado)
                self.assertGreaterEqual(self._contraste(ajustado, fondo), 3)
                self.assertGreater(max(canales) - min(canales), 15)
                self.assertNotIn(ajustado.lower(), ("#000000", "#ffffff"))

    def test_colores_css_con_nombre_no_se_ajustan_si_tienen_contraste(self):
        for oscuro, fondo in (
            (False, streamlit_app.PALETA["superficie"]),
            (True, streamlit_app.PALETA_OSCURA["superficie"]),
        ):
            with self.subTest(oscuro=oscuro):
                figura = go.Figure(
                    go.Scatter(x=[0], y=[1], mode="markers", marker={"color": "gray"})
                )
                streamlit_app._aplicar_tema_figura(figura, oscuro)

                color = figura.data[0].marker.color
                self.assertEqual(color.lower(), "gray")
                self.assertGreaterEqual(self._contraste(color, fondo), 3)

    def test_etiquetas_de_heatmap_usen_blanco_y_negro_segun_la_celda(self):
        for oscuro in (False, True):
            with self.subTest(oscuro=oscuro):
                themed = px.imshow(
                    pd.DataFrame([[0.0, 0.5, 1.0]]),
                    text_auto=".2f",
                    color_continuous_scale="RdBu",
                    zmin=0,
                    zmax=1,
                )
                streamlit_app._aplicar_tema_figura(themed, oscuro)
                etiquetas = next(
                    traza for traza in themed.data
                    if traza.meta == "plotly-theme-cell-labels"
                )
                self.assertEqual(len(etiquetas.text), 3)
                self.assertEqual(
                    set(etiquetas.textfont.color),
                    {"#172019", "#ffffff"},
                )
                self.assertEqual(themed.data[0].z.tolist(), [[0.0, 0.5, 1.0]])
                self.assertFalse(themed.data[0].texttemplate)
                self.assertEqual(etiquetas.texttemplate, "%{text:.2f}")

    def test_heatmaps_grandes_conservan_todas_las_etiquetas(self):
        valores = [[float(fila + columna) for columna in range(25)] for fila in range(25)]
        figura = px.imshow(valores, text_auto=",.1f", color_continuous_scale="Viridis")
        streamlit_app._aplicar_tema_figura(figura, oscuro=True)

        etiquetas = next(
            traza for traza in figura.data
            if traza.meta == "plotly-theme-cell-labels"
        )
        self.assertEqual(len(etiquetas.text), 625)
        self.assertEqual(etiquetas.texttemplate, "%{text:,.1f}")
        self.assertEqual(len(etiquetas.textfont.color), 625)

    def test_colores_adaptativos_respetan_coloraxis_reverse_y_cmid(self):
        figura = px.imshow(
            [[0.0, 7.0, 10.0]],
            text_auto=".1f",
            color_continuous_scale=["#000000", "#ffffff"],
            zmin=0,
            zmax=10,
        )
        figura.update_coloraxes(cmid=7, reversescale=True)
        streamlit_app._aplicar_tema_figura(figura, oscuro=False)

        etiquetas = figura.data[-1]
        colores = list(etiquetas.textfont.color)
        self.assertEqual(colores[0], "#172019")
        self.assertEqual(colores[1], "#ffffff")
        self.assertEqual(colores[-1], "#ffffff")

        automatica = px.imshow(
            [[-1.0, 0.0, 10.0]],
            text_auto=".1f",
            color_continuous_scale=["#000000", "#ffffff"],
        )
        automatica.update_coloraxes(cmid=0, cauto=True)
        streamlit_app._aplicar_tema_figura(automatica, oscuro=False)
        colores = list(automatica.data[-1].textfont.color)
        self.assertEqual(colores[1], "#172019")

    def test_fabricas_reales_de_heatmap_reciben_etiquetas_tematicas(self):
        datos = EDA(dataframe=pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}))
        figura_correlacion, _ = datos.mapa_calor(mostrar=False)
        perfiles = SimpleNamespace(
            centroides=pd.DataFrame([[0.1, -0.9], [0.8, 0.2]], columns=["A", "B"]),
            algoritmo="KMeans",
        )
        confusion = SimpleNamespace(
            matriz_confusion=pd.DataFrame([[4, 1], [2, 5]]), algoritmo="RF"
        )
        figuras = (
            figura_correlacion,
            VisualizadorNoSupervisado.perfiles_cluster(perfiles),
            VisualizadorSupervisado.matriz_confusion(confusion),
        )
        for figura in figuras:
            with self.subTest(titulo=figura.layout.title.text):
                streamlit_app._aplicar_tema_figura(figura, oscuro=True)
                self.assertTrue(
                    any(
                        traza.meta == "plotly-theme-cell-labels"
                        for traza in figura.data
                    )
                )

    def test_heatmap_real_redondeado_json_conserva_etiquetas_y_datos(self):
        datos = EDA(dataframe=pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}))
        figura, matriz = datos.mapa_calor(mostrar=False)
        serializada = pio.from_json(pio.to_json(figura))
        z_original = serializada.data[0].z.copy()
        self.assertIsInstance(z_original, dict)

        for oscuro in (False, True):
            with self.subTest(oscuro=oscuro):
                themed = pio.from_json(pio.to_json(figura))
                z_original = themed.data[0].z.copy()
                streamlit_app._aplicar_tema_figura(themed, oscuro)

                etiquetas = next(
                    traza for traza in themed.data
                    if traza.meta == "plotly-theme-cell-labels"
                )
                self.assertEqual(len(etiquetas.text), matriz.size)
                self.assertEqual(
                    list(etiquetas.text),
                    [str(valor) for valor in matriz.to_numpy().ravel()],
                )
                self.assertTrue(
                    set(etiquetas.textfont.color) <= {"#172019", "#ffffff"}
                )
                self.assertTrue(etiquetas.textfont.color)
                self.assertEqual(themed.data[0].z, z_original)

    def test_figura_pie_de_tipos_conserva_sectores_y_etiquetas_legibles(self):
        for oscuro, fondo in (
            (False, streamlit_app.PALETA["superficie"]),
            (True, streamlit_app.PALETA_OSCURA["superficie"]),
        ):
            with self.subTest(oscuro=oscuro):
                figura = VisualizadorDatos.tipos_columnas(
                    pd.DataFrame({"tipo": ["Numérica", "Categórica", "Numérica"]})
                )
                traza = figura.data[0]
                etiquetas, valores = list(traza.labels), list(traza.values)
                streamlit_app._aplicar_tema_figura(figura, oscuro)

                colores = list(traza.marker.colors)
                textos = list(traza.textfont.color)
                self.assertEqual(list(traza.labels), etiquetas)
                self.assertEqual(list(traza.values), valores)
                self.assertEqual(len(set(colores)), 2)
                self.assertEqual(len(textos), len(colores))
                for color, texto in zip(colores, textos):
                    self.assertGreaterEqual(self._contraste(color, fondo), 3)
                    self.assertGreaterEqual(self._contraste(texto, color), 4.5)
                    self.assertNotIn(color.lower(), ("#000000", "#ffffff"))

    def test_figura_pie_se_mantiene_legible_al_cambiar_de_tema(self):
        figura = VisualizadorDatos.tipos_columnas(
            pd.DataFrame({"tipo": ["Numérica", "Categórica"]})
        )
        for oscuro in (True, False, True):
            with self.subTest(oscuro=oscuro):
                streamlit_app._aplicar_tema_figura(figura, oscuro)
                fondo = (streamlit_app.PALETA_OSCURA if oscuro else streamlit_app.PALETA)["superficie"]
                traza = figura.data[0]
                self.assertEqual(len(set(traza.marker.colors)), 2)
                for color, texto in zip(traza.marker.colors, traza.textfont.color):
                    self.assertGreaterEqual(self._contraste(color, fondo), 3)
                    self.assertGreaterEqual(self._contraste(texto, color), 4.5)

    @staticmethod
    def _contraste(primer_color, segundo_color):
        def luminancia(color):
            canales = streamlit_app._color_rgb(color)
            if canales is None:
                canales = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
            lineales = [
                canal / 12.92
                if canal / 255 <= 0.04045
                else ((canal / 255 + 0.055) / 1.055) ** 2.4
                for canal in canales
            ]
            return 0.2126 * lineales[0] + 0.7152 * lineales[1] + 0.0722 * lineales[2]

        bajo, alto = sorted((luminancia(primer_color), luminancia(segundo_color)))
        return (alto + 0.05) / (bajo + 0.05)

    def test_renderizador_selecciona_tema_y_llama_plotly_chart(self):
        figura = go.Figure(go.Scatter(x=[0], y=[1], marker={"color": "#1a3c2b"}))
        contexto = SimpleNamespace(theme=SimpleNamespace(type="dark"))
        with (
            patch.object(streamlit_app.st, "context", contexto),
            patch.object(streamlit_app.st, "plotly_chart") as renderizar,
        ):
            streamlit_app._mostrar_figura(figura)

        self.assertEqual(figura.layout.paper_bgcolor, "#111713")
        renderizar.assert_called_once_with(figura, width="stretch")


if __name__ == "__main__":
    unittest.main()
