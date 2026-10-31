"""Vista del agente EDA en modo headless; el LLM se sustituye por ProveedorFalso."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from framework_ia.agentes import ProveedorFalso, RespuestaLLM
from framework_ia.utils import PROVEEDOR_SIN_LLM

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


def datos():
    rng = np.random.default_rng(5)
    x = rng.normal(size=50)
    return pd.DataFrame(
        {"id": np.arange(50), "x": x, "ruido": rng.normal(size=50),
         "y": 3 * x + 2 * rng.normal(size=50)}
    ).assign(fuga=lambda d: d["y"] * 2 + 1)


def nueva_app():
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (datos(), "agentes.csv", "agentes-ui")
    app.run()
    app.sidebar.button(key="nav_eda_agente").click().run()
    assert not app.exception
    return app


def iniciar(app, proveedor=PROVEEDOR_SIN_LLM):
    app.selectbox(key="agentes_target").set_value("y")
    app.selectbox(key="agentes_proveedor").set_value(proveedor)
    app.button[0].click()  # el envío del formulario es el primer botón del cuerpo
    app.run()
    assert not app.exception
    return app


def test_agente_esta_en_exploracion_y_la_ruta_antigua_redirige():
    app = nueva_app()
    assert any("Exploración / Agente IA" in m.value for m in app.markdown)
    assert not [b for b in app.sidebar.button if b.key == "nav_agentes"]
    assert "agentes_sesion" not in app.session_state  # navegar no analiza

    app.session_state["vista_activa"] = "agentes"
    app.run()
    assert not app.exception
    assert app.session_state["vista_activa"] == "eda_agente"


def test_sin_llm_muestra_hallazgos_y_aplica_sugerencia():
    app = iniciar(nueva_app())

    assert any("identificador" in aviso.value for aviso in app.warning)
    sesion = app.session_state["agentes_sesion"]
    assert sesion["agente"].llm is None
    assert "api_key" not in sesion["opciones"]
    assert sesion["pizarra"].origen == "agentes.csv"  # sin ruta local hacia el LLM

    app.button(key="agentes_aplicar").click().run()
    assert app.session_state["vista_activa"] == "configuracion"
    assert app.session_state["sup_tarea"] == "regresion"
    assert app.session_state["sup_target"] == "y"
    assert app.session_state["sup_features"] == ["x", "ruido"]


def test_con_llm_falso_interpreta_y_responde_preguntas():
    llm = ProveedorFalso([RespuestaLLM(texto="Interpretación inicial"), RespuestaLLM(texto="Sin nulos.")])
    with patch("framework_ia.ui.agentes.crear_proveedor", return_value=llm):
        app = iniciar(nueva_app(), proveedor="gemini")
        assert "Interpretación inicial" in [m.markdown[0].value for m in app.chat_message]

        app.chat_input(key="agentes_pregunta").set_value("¿Hay nulos?").run()
    textos = [m.markdown[0].value for m in app.chat_message]
    assert textos[-2:] == ["¿Hay nulos?", "Sin nulos."]


def test_error_del_proveedor_se_muestra():
    with patch("framework_ia.ui.agentes.crear_proveedor", side_effect=ValueError("Defina GEMINI_API_KEY")):
        app = iniciar(nueva_app(), proveedor="gemini")

    assert any("GEMINI_API_KEY" in error.value for error in app.error)
    assert "agentes_sesion" not in app.session_state


def test_cambiar_de_dataset_descarta_la_sesion_de_agentes():
    app = iniciar(nueva_app())
    assert "agentes_sesion" in app.session_state

    app.session_state["fuente_aplicada"] = (datos().iloc[:30], "otro.csv", "otro")
    app.run()
    assert "agentes_sesion" not in app.session_state
