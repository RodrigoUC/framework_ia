"""Estilos CSS de la aplicación, cálculo de contraste y renderizado de figuras Plotly."""

from __future__ import annotations

import base64
import math
import re
from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from framework_ia.ui.modelos.catalogo_vistas import PALETA, PALETA_OSCURA

MostrarFigura = Callable[..., None]


def _aplicar_estilos_atlas() -> None:
    """Instala la hoja de estilos compartida de la interfaz según el tema activo."""
    oscuro = st.context.theme.type == "dark"
    focus_color = "#9effbf" if oscuro else "#1a3c2b"
    chip_background = "#9effbf" if oscuro else "#1a3c2b"
    chip_foreground = "#111713" if oscuro else "#f7f7f5"
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap');
        @import url('https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600&display=swap');

        .stApp { font-family: 'General Sans', sans-serif; }
        :root { --atlas-focus-color: __ATLAS_FOCUS_COLOR__; }
        .stApp::selection { background: #9effbf; color: #1a3c2b; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
        code, [data-testid="stCaptionContainer"] { font-family: 'JetBrains Mono', monospace; }
        [data-testid="stSidebar"] { border-right: 1px solid color-mix(in srgb, currentColor 28%, transparent); }
        [data-testid="stSidebar"] div.stButton, [data-testid="stSidebar"] div[data-testid="stButton"], [data-testid="stSidebar"] .stButton { width: 100% !important; margin-bottom: .35rem; }
        [data-testid="stSidebar"] .stButton > button { width: 100% !important; min-height: 2.35rem; justify-content: flex-start; border: 1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius: 2px; font-family: 'General Sans', sans-serif; font-size: .88rem; padding: 0 .95rem; box-sizing: border-box; }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] { border-left: 2px solid var(--primary-color); }
        [data-testid="stSidebar"] details { border: 1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius: 2px; background: color-mix(in srgb, currentColor 4%, transparent); margin-bottom: .35rem; }
        [data-testid="stSidebar"] summary { font-family: 'General Sans', sans-serif; font-size: .88rem; min-height: 2.35rem; display: flex; align-items: center; padding: 0 .95rem; box-sizing: border-box; }
        .atlas-nav-label { margin: 1.35rem 0 .35rem; color: color-mix(in srgb, currentColor 72%, transparent); font-family: 'JetBrains Mono', monospace; font-size: .67rem; font-weight: 500; letter-spacing: .14em; text-transform: uppercase; }
        .atlas-family { margin: .7rem 0 .15rem; color: color-mix(in srgb, currentColor 72%, transparent); font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; }
        .atlas-header { margin: .25rem 0 1.25rem; padding: 1.35rem 1.55rem; border: 1px solid color-mix(in srgb, currentColor 28%, transparent); background: transparent; }
        .atlas-breadcrumb { color: inherit; opacity: .82; font-family: 'JetBrains Mono', monospace; font-size: .68rem; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; }
        .atlas-title { margin: .45rem 0 .25rem; color: inherit; font-family: 'Space Grotesk', sans-serif; font-size: 2.15rem; font-weight: 600; letter-spacing: -.03em; }
        .atlas-subtitle { margin: 0; color: inherit; opacity: .78; font-size: .94rem; }
        [data-testid="stMetric"] { border: 1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius: 2px; background: color-mix(in srgb, currentColor 2%, transparent); padding: .28rem .55rem; min-height: auto; }
        [data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: .60rem; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 0 !important; }
        [data-testid="stMetricLabel"] p { font-size: .60rem !important; margin: 0 !important; line-height: 1.15 !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.12rem !important; line-height: 1.2 !important; font-weight: 600; }
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) { gap: .45rem; margin-bottom: .35rem; }
        .stButton > button[kind="primary"] { border-radius: 2px; }
        .stButton > button:focus-visible, input:focus-visible { outline: 2px solid var(--atlas-focus-color) !important; outline-offset: 2px; }
        [data-testid="stSidebar"] .stButton > button:focus-visible, [data-testid="stSidebar"] input:focus-visible { outline-color: #f4d35e !important; }
        [data-testid="stDataFrame"] { border: 1px solid color-mix(in srgb, currentColor 28%, transparent); }
        [data-testid="stMultiSelect"] [data-tag] { background-color: __ATLAS_CHIP_BACKGROUND__; color: __ATLAS_CHIP_FOREGROUND__; }
        [data-testid="stMultiSelect"] [data-tag] *, [data-testid="stMultiSelect"] [data-tag] svg { color: __ATLAS_CHIP_FOREGROUND__; fill: __ATLAS_CHIP_FOREGROUND__; }
        @media (max-width: 900px) { .atlas-header { padding: 1rem; } .atlas-title { font-size: 1.65rem; } }
        </style>
        """.replace("__ATLAS_FOCUS_COLOR__", focus_color)
        .replace("__ATLAS_CHIP_BACKGROUND__", chip_background)
        .replace("__ATLAS_CHIP_FOREGROUND__", chip_foreground),
        unsafe_allow_html=True,
    )


def _color_rgb(color: str) -> tuple[int, int, int] | None:
    """Convierte colores hexadecimales o rgb() a canales RGB."""
    if not isinstance(color, str):
        return None
    color = color.strip()
    if color.startswith("#") and len(color) in {4, 7}:
        if len(color) == 4:
            color = "#" + "".join(channel * 2 for channel in color[1:])
        try:
            return tuple(
                int(color[index : index + 2], 16) for index in (1, 3, 5)
            )
        except ValueError:
            return None
    if color.startswith("rgb(") and color.endswith(")"):
        try:
            return tuple(
                int(float(channel.strip()))
                for channel in color[4:-1].split(",")[:3]
            )
        except ValueError:
            return None
    colores_css = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "silver": (192, 192, 192),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "lime": (0, 255, 0),
        "navy": (0, 0, 128),
        "teal": (0, 128, 128),
        "olive": (128, 128, 0),
        "maroon": (128, 0, 0),
    }
    return colores_css.get(color.lower())


def _luminancia(color: str) -> float | None:
    """Calcula la luminancia relativa sRGB para etiquetas sobre celdas Plotly."""
    canales = _color_rgb(color)
    if canales is None:
        return None
    lineales = [
        canal / 12.92
        if canal / 255 <= 0.04045
        else ((canal / 255 + 0.055) / 1.055) ** 2.4
        for canal in canales
    ]
    return 0.2126 * lineales[0] + 0.7152 * lineales[1] + 0.0722 * lineales[2]


def _color_texto_celda(color: str) -> str:
    """Elige negro o blanco para mantener contraste en cada celda del mapa."""
    luminancia = _luminancia(color)
    if luminancia is None:
        return "#172019"
    contraste_oscuro = (luminancia + 0.05) / 0.05
    contraste_claro = 1.05 / (luminancia + 0.05)
    return "#172019" if contraste_oscuro >= contraste_claro else "#ffffff"


def _color_texto_sector(color: str, predeterminado: str) -> str:
    """Elige texto AA para una sección circular con color explícito."""
    luminancia = _luminancia(color)
    if luminancia is None:
        return predeterminado
    return "#000000" if (luminancia + 0.05) / 0.05 >= 4.5 else "#ffffff"


def _ajustar_color_traza(color, oscuro: bool):
    """Eleva el contraste de colores de marcas y líneas sin perder su matiz."""
    if isinstance(color, (list, tuple)):
        return [_ajustar_color_traza(valor, oscuro) for valor in color]
    canales = _color_rgb(color)
    if canales is None:
        return color
    fondo = PALETA_OSCURA["superficie"] if oscuro else PALETA["superficie"]
    luminancia_fondo = _luminancia(fondo)

    def contraste(luminancia: float) -> float:
        menor, mayor = sorted((luminancia, luminancia_fondo))
        return (mayor + 0.05) / (menor + 0.05)

    luminancia = _luminancia(color)
    if luminancia is None or contraste(luminancia) >= 3:
        return color
    destino = (255, 255, 255) if oscuro else (0, 0, 0)
    for paso in range(1, 101):
        proporcion = paso / 100
        ajustado = tuple(
            round(canal + (meta - canal) * proporcion)
            for canal, meta in zip(canales, destino)
        )
        hexadecimal = "#" + "".join(f"{canal:02x}" for canal in ajustado)
        if contraste(_luminancia(hexadecimal)) >= 3:
            return hexadecimal
    return "#ffffff" if oscuro else "#000000"


def _texto_mapa_calor(figura: go.Figure) -> list[go.Scatter]:
    """Superpone etiquetas por celda con colores contrastantes y formato Plotly."""
    etiquetas = []
    for traza in figura.data:
        if traza.type != "heatmap" or not traza.texttemplate:
            continue
        valores = traza.z
        if isinstance(valores, dict):
            try:
                forma = tuple(
                    int(dimension) for dimension in valores["shape"].split(",")
                )
                arreglo = np.frombuffer(
                    base64.b64decode(valores["bdata"], validate=True),
                    dtype=np.dtype(valores["dtype"]),
                )
                if not forma or arreglo.size != math.prod(forma):
                    continue
                valores = arreglo.reshape(forma).tolist()
            except (KeyError, TypeError, ValueError):
                continue
        if valores is None or not len(valores) or not len(valores[0]):
            continue
        coloraxis = getattr(traza, "coloraxis", None)
        eje_color = getattr(figura.layout, coloraxis, None) if coloraxis else None
        escala = (
            (eje_color.colorscale if eje_color else None)
            or traza.colorscale
            or "Viridis"
        )
        minimo = eje_color.cmin if eje_color else traza.zmin
        maximo = eje_color.cmax if eje_color else traza.zmax
        centro = eje_color.cmid if eje_color else traza.zmid
        rango_automatico = minimo is None and maximo is None
        invertir = bool(
            (eje_color and eje_color.reversescale) or traza.reversescale
        )
        valores_validos = [
            float(valor)
            for fila in valores
            for valor in fila
            if valor is not None and math.isfinite(float(valor))
        ]
        if not valores_validos:
            continue
        minimo = minimo if minimo is not None else min(valores_validos)
        maximo = maximo if maximo is not None else max(valores_validos)
        if centro is not None and rango_automatico:
            radio = max(abs(minimo - centro), abs(maximo - centro))
            minimo, maximo = centro - radio, centro + radio
        xs = (
            list(traza.x)
            if traza.x is not None
            else list(range(len(valores[0])))
        )
        ys = list(traza.y) if traza.y is not None else list(range(len(valores)))
        xs_repetidos, ys_repetidos, celdas, posiciones = [], [], [], []
        if not isinstance(escala, str):
            escala = [list(punto) for punto in escala]
        for fila, y in zip(valores, ys):
            for valor, x in zip(fila, xs):
                if valor is None or not math.isfinite(float(valor)):
                    continue
                posicion = (
                    0.5
                    if maximo == minimo
                    else min(
                        1,
                        max(0, (float(valor) - minimo) / (maximo - minimo)),
                    )
                )
                if (
                    centro is not None
                    and rango_automatico
                    and minimo < centro < maximo
                ):
                    posicion = (
                        0.5 * (float(valor) - minimo) / (centro - minimo)
                        if float(valor) <= centro
                        else 0.5
                        + 0.5 * (float(valor) - centro) / (maximo - centro)
                    )
                if invertir:
                    posicion = 1 - posicion
                xs_repetidos.append(x)
                ys_repetidos.append(y)
                celdas.append(valor)
                posiciones.append(posicion)
        colores = [
            _color_texto_celda(color)
            for color in sample_colorscale(escala, posiciones)
        ]
        plantilla = re.sub(r"(%\{)z(?=[:}])", r"\1text", traza.texttemplate)
        etiquetas.append(
            go.Scatter(
                x=xs_repetidos,
                y=ys_repetidos,
                text=celdas,
                mode="text",
                texttemplate=plantilla,
                textfont={"color": colores, "size": 12},
                hoverinfo="skip",
                showlegend=False,
                meta="plotly-theme-cell-labels",
            )
        )
        traza.texttemplate = ""
    return etiquetas


def _aplicar_tema_figura(figura: go.Figure, oscuro: bool) -> None:
    """Alinea fondo, ejes, trazas y anotaciones con el tema Streamlit activo."""
    figura.update_layout(
        template="plotly_dark" if oscuro else "plotly_white",
        paper_bgcolor=PALETA_OSCURA["fondo"] if oscuro else PALETA["fondo"],
        plot_bgcolor=PALETA_OSCURA["superficie"]
        if oscuro
        else PALETA["superficie"],
        font_color=PALETA_OSCURA["texto"] if oscuro else PALETA["texto"],
        margin=dict(l=40, r=20, t=60, b=40),
    )
    color_texto = PALETA_OSCURA["texto"] if oscuro else PALETA["texto"]
    color_grid = "#46534a" if oscuro else "#d2d3cd"
    figura.update_xaxes(
        color=color_texto,
        gridcolor=color_grid,
        zerolinecolor=color_grid,
        linecolor=color_grid,
    )
    figura.update_yaxes(
        color=color_texto,
        gridcolor=color_grid,
        zerolinecolor=color_grid,
        linecolor=color_grid,
    )
    figura.update_annotations(font_color=color_texto, arrowcolor=color_texto)
    for forma in figura.layout.shapes or ():
        if forma.line.color:
            forma.line.color = _ajustar_color_traza(forma.line.color, oscuro)
    for traza in figura.data:
        for atributo in ("marker", "line"):
            estilo = getattr(traza, atributo, None)
            color = getattr(estilo, "color", None) if estilo is not None else None
            if color is not None:
                estilo.color = _ajustar_color_traza(color, oscuro)
            if estilo is not None and getattr(estilo, "colors", None) is not None:
                estilo.colors = _ajustar_color_traza(estilo.colors, oscuro)
        if getattr(traza, "textfont", None) is not None and not (
            isinstance(traza.meta, str)
            and traza.meta == "plotly-theme-cell-labels"
        ):
            if traza.type == "pie" and traza.marker.colors:
                traza.textfont.color = [
                    _color_texto_sector(color, color_texto)
                    for color in traza.marker.colors
                ]
            else:
                traza.textfont.color = color_texto
    for traza in _texto_mapa_calor(figura):
        figura.add_trace(traza)


def _mostrar_figura(
    figura: go.Figure, key: str | None = None, *, alt: str | None = None
) -> None:
    """Renderiza una figura Plotly alineada con el tema claro/oscuro activo."""
    _aplicar_tema_figura(figura, st.context.theme.type == "dark")
    opciones = {"width": "stretch"}
    if alt is not None:
        st.caption(alt)
    if key is not None:
        opciones["key"] = key


    st.plotly_chart(figura, **opciones)
