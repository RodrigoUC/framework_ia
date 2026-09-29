"""Regression tests for accessible Streamlit primary-button theme tokens."""

import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from framework_ia.ui import streamlit_app


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

    def test_primary_button_fill_text_and_hover_remain_owned_by_theme(self):
        for theme_name in ("light", "dark"):
            with self.subTest(theme=theme_name):
                context = SimpleNamespace(theme=SimpleNamespace(type=theme_name))
                with (
                    patch.object(streamlit_app.st, "context", context),
                    patch.object(streamlit_app.st, "markdown") as markdown,
                ):
                    streamlit_app._aplicar_estilos_atlas()

                stylesheet = markdown.call_args.args[0]
                self.assertNotIn("#122d20", stylesheet)
                self.assertNotIn('button[kind="primary"]:hover', stylesheet)

                primary_rules = (
                    rule.split("}", 1)[0]
                    for rule in stylesheet.split("{")
                    if 'button[kind="primary"]' in rule
                )
                for rule in primary_rules:
                    self.assertNotIn("background", rule)
                    self.assertNotIn("color:", rule)

        # The stylesheet no longer overrides Streamlit's native button colors;
        # both configured primary tokens contrast with white text.
        for theme_name, primary in (
            ("light", self.theme["primaryColor"]),
            ("dark", self.theme["dark"]["primaryColor"]),
        ):
            with self.subTest(theme=theme_name, state="normal-and-hover-token"):
                self.assertGreaterEqual(_contrast(primary, "#ffffff"), 4.5)


if __name__ == "__main__":
    unittest.main()
