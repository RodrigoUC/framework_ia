"""Utilidades de color para gráficos."""

from __future__ import annotations


def color_texto_por_luminancia(color) -> str:
    """Elige texto legible sobre un color de celda RGBA."""

    def luminancia_rgb(rgb) -> float:
        canales = [
            canal / 12.92 if canal <= 0.04045 else ((canal + 0.055) / 1.055) ** 2.4
            for canal in rgb
        ]
        return 0.2126 * canales[0] + 0.7152 * canales[1] + 0.0722 * canales[2]

    luminancia_celda = luminancia_rgb(color[:3])
    candidatos = {"#000000": (0.0, 0.0, 0.0), "#ffffff": (1.0, 1.0, 1.0)}
    return max(
        candidatos,
        key=lambda foreground: (
            max(luminancia_rgb(candidatos[foreground]), luminancia_celda) + 0.05
        )
        / (min(luminancia_rgb(candidatos[foreground]), luminancia_celda) + 0.05),
    )
