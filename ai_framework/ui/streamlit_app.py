"""Streamlit app entrypoint compatibility redirect."""

from __future__ import annotations

import streamlit as st

from .app import *
from ai_framework.datasets.diabetes import cargar_diabetes_regresion
from ai_framework.ui.controllers.sources_controller import (
    _aplicar_preset_regresion,
    _cargar_fuente_csv,
    _cargar_referencia_regresion,
    _guardar_metadatos_fuente,
    _limpiar_resultados,
    _sincronizar_dataset,
)
from ai_framework.ui.views.theme import (
    _ajustar_color_traza,
    _aplicar_estilos_atlas,
    _aplicar_tema_figura,
    _color_rgb,
    _color_texto_celda,
    _color_texto_sector,
    _luminancia,
    _mostrar_figura,
    _texto_mapa_calor,
)
from ai_framework.ui.views.data_view import (
    _obtener_base_dir,
    _obtener_cargador_csv,
    _seleccionar_fuente,
)
