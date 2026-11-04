"""Color utilities for charts and plots."""

from __future__ import annotations


def rgb_luminance(rgb) -> float:
    canales = [
        canal / 12.92 if canal <= 0.04045 else ((canal + 0.055) / 1.055) ** 2.4
        for canal in rgb
    ]
    return 0.2126 * canales[0] + 0.7152 * canales[1] + 0.0722 * canales[2]


def text_color_by_luminance(color) -> str:
    """Return readable black or white text color over a given RGBA background."""
    luminancia_celda = rgb_luminance(color[:3])
    candidatos = {"#000000": (0.0, 0.0, 0.0), "#ffffff": (1.0, 1.0, 1.0)}
    return max(
        candidatos,
        key=lambda foreground: (
            max(rgb_luminance(candidatos[foreground]), luminancia_celda) + 0.05
        )
        / (min(rgb_luminance(candidatos[foreground]), luminancia_celda) + 0.05),
    )


color_texto_por_luminancia = text_color_by_luminance
luminancia_rgb = rgb_luminance
