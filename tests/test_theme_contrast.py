"""Focused tests for theme-independent heatmap annotation contrast."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd

from framework_ia.datos.dataframe import DataFrame, _color_texto_por_luminancia
from framework_ia.ui import streamlit_app
from framework_ia.visualizacion import VisualizadorNoSupervisado


def _relative_luminance(rgb) -> float:
    channels = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in rgb[:3]
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first, second) -> float:
    luminances = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (luminances[1] + 0.05) / (luminances[0] + 0.05)


def _hex_rgb(value: str):
    return tuple(int(value[index : index + 2], 16) / 255 for index in (1, 3, 5))


class HeatmapAnnotationContrastTests(unittest.TestCase):
    def test_foreground_meets_wcag_contrast_across_cmap_samples(self):
        samples = [
            colormaps["vlag"](position)
            for position in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        samples.extend(colormaps["viridis"](position) for position in (0, 0.5, 1))

        for cell_color in samples:
            with self.subTest(cell_color=cell_color):
                foreground = _color_texto_por_luminancia(cell_color)
                self.assertGreaterEqual(
                    _contrast_ratio(_hex_rgb(foreground), cell_color), 4.5
                )

    def test_cell_luminance_selects_both_annotation_foregrounds(self):
        selected = {
            _color_texto_por_luminancia(colormaps["vlag"](position))
            for position in (0.0, 0.5, 1.0)
        }
        self.assertIn("#000000", selected)
        self.assertIn("#ffffff", selected)

    def test_correlation_heatmap_annotations_use_both_foregrounds(self):
        first = pd.Series([1, -1, 1, -1], dtype=float)
        second = pd.Series([1, 1, -1, -1], dtype=float)
        data = pd.DataFrame(
            {"first": first, "second": second, "mixed": first + second}
        )
        frame = DataFrame(data)
        frame.mapa_calor(mostrar=False)
        figure = plt.gcf()
        try:
            colors = {
                annotation.get_color()
                for axis in figure.axes
                for annotation in axis.texts
            }
            self.assertIn("#000000", colors)
            self.assertIn("#ffffff", colors)
        finally:
            plt.close(figure)


class ThemeRenderingContrastTests(unittest.TestCase):
    def test_multiselect_chip_palette_meets_wcag_in_both_themes(self):
        with patch.object(streamlit_app.st, "markdown") as markdown:
            streamlit_app._aplicar_estilos_atlas()

        stylesheet = markdown.call_args.args[0]
        self.assertIn('[data-testid="stMultiSelect"] [data-tag]', stylesheet)
        self.assertIn("background-color: #1a3c2b", stylesheet)
        self.assertIn("color: #f7f7f5", stylesheet)
        self.assertNotIn("        [data-tag] {", stylesheet)
        ratio = _contrast_ratio(_hex_rgb("#f7f7f5"), _hex_rgb("#1a3c2b"))
        self.assertGreaterEqual(ratio, 4.5)

    def test_figure_background_and_text_follow_both_active_themes(self):
        palettes = {
            "light": (streamlit_app.PALETA, "#f7f7f5", "#eeeee9", "#242523"),
            "dark": (streamlit_app.PALETA_OSCURA, "#111713", "#1a241d", "#f7f7f5"),
        }
        for theme, (_, background, surface, text) in palettes.items():
            with self.subTest(theme=theme):
                figure, axis = plt.subplots()
                annotation = axis.text(0.5, 0.5, "heatmap value", color="#aabbcc")
                fake_context = SimpleNamespace(
                    theme=SimpleNamespace(type=theme)
                )
                try:
                    with patch.object(streamlit_app.st, "context", fake_context):
                        streamlit_app._aplicar_tema_oscuro_a_figura(figure)
                    self.assertEqual(
                        figure.get_facecolor(), _hex_rgb(background) + (1.0,)
                    )
                    self.assertEqual(
                        axis.get_facecolor(), _hex_rgb(surface) + (1.0,)
                    )
                    self.assertEqual(axis.title.get_color(), text)
                    self.assertEqual(annotation.get_color(), text)
                finally:
                    plt.close(figure)

    def test_acp_labels_follow_dark_theme_without_changing_heatmap_cells(self):
        coordinates = pd.DataFrame(
            {"CP1": [-1.0, 0.0, 1.0], "CP2": [0.5, -1.0, 0.5]}
        )
        loads = pd.DataFrame(
            {"CP1": [0.7, -0.4], "CP2": [0.4, 0.8]},
            index=["feature_a", "feature_b"],
        )
        acp = SimpleNamespace(coordenadas=coordinates, cargas=loads)
        fake_context = SimpleNamespace(theme=SimpleNamespace(type="dark"))

        for renderer in (
            VisualizadorNoSupervisado.circulo_correlacion,
            VisualizadorNoSupervisado.sobreposicion_acp,
        ):
            with self.subTest(renderer=renderer.__name__):
                figure = renderer(acp)
                try:
                    with patch.object(streamlit_app.st, "context", fake_context):
                        streamlit_app._aplicar_tema_oscuro_a_figura(figure)
                    labels = [
                        text.get_color()
                        for axis in figure.axes
                        for text in axis.texts
                        if text.get_text().startswith("feature_")
                    ]
                    self.assertEqual(labels, ["#f7f7f5", "#f7f7f5"])
                finally:
                    plt.close(figure)

        frame = DataFrame(
            pd.DataFrame({"first": [1, -1, 1, -1], "second": [1, 1, -1, -1]})
        )
        frame.mapa_calor(mostrar=False)
        figure = plt.gcf()
        try:
            annotations_before = [
                text.get_color() for axis in figure.axes for text in axis.texts
            ]
            with patch.object(streamlit_app.st, "context", fake_context):
                streamlit_app._aplicar_tema_oscuro_a_figura(figure)
            annotations_after = [
                text.get_color() for axis in figure.axes for text in axis.texts
            ]
            self.assertEqual(annotations_after, annotations_before)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
