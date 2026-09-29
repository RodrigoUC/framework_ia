"""Focused tests for theme-independent heatmap annotation contrast."""

import unittest

import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd

from framework_ia.datos.dataframe import DataFrame, _color_texto_por_luminancia


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


if __name__ == "__main__":
    unittest.main()
