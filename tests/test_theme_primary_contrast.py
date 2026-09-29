"""Regression tests for accessible Streamlit primary-button theme tokens."""

import tomllib
import unittest
from pathlib import Path


THEME_CONFIG = Path(__file__).parents[1] / ".streamlit" / "config.toml"


def _luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


class PrimaryThemeContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with THEME_CONFIG.open("rb") as config_file:
            cls.theme = tomllib.load(config_file)["theme"]

    def test_dark_main_and_sidebar_primary_tokens_are_accessible(self):
        dark = self.theme["dark"]
        cases = (
            (
                "main",
                dark["primaryColor"],
                dark["backgroundColor"],
                dark["textColor"],
            ),
            (
                "sidebar",
                dark["sidebar"]["primaryColor"],
                dark["sidebar"]["backgroundColor"],
                dark["sidebar"]["textColor"],
            ),
        )
        self.assertEqual(cases[0][1], cases[1][1])
        for location, primary, background, text_color in cases:
            with self.subTest(location=location):
                self.assertGreaterEqual(_contrast(primary, "#ffffff"), 4.5)
                self.assertGreaterEqual(_contrast(primary, text_color), 4.5)
                self.assertGreaterEqual(_contrast(primary, background), 3)

    def test_light_main_and_sidebar_primary_tokens_remain_accessible(self):
        light = self.theme["light"]
        cases = (
            ("main", self.theme["primaryColor"], light["backgroundColor"]),
            (
                "sidebar",
                light["sidebar"]["primaryColor"],
                light["sidebar"]["backgroundColor"],
            ),
        )
        for location, primary, background in cases:
            with self.subTest(location=location):
                self.assertGreaterEqual(_contrast(primary, "#ffffff"), 4.5)
                self.assertGreaterEqual(_contrast(primary, background), 3)


if __name__ == "__main__":
    unittest.main()
