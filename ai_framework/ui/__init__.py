"""Streamlit UI package structured in MVC and Clean Architecture."""

from __future__ import annotations

import sys
from . import app, controllers, i18n, models, streamlit_app, views

controladores = controllers
modelos = models
vistas = views

sys.modules.setdefault("ai_framework.ui.controladores", controllers)
sys.modules.setdefault("ai_framework.ui.modelos", models)
sys.modules.setdefault("ai_framework.ui.vistas", views)

__all__ = [
    "app",
    "controllers",
    "controladores",
    "i18n",
    "models",
    "modelos",
    "streamlit_app",
    "views",
    "vistas",
]
