"""EDA agent view in headless mode; the LLM is replaced by FakeLLMProvider."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ai_framework.agents import FakeLLMProvider, LLMResponse
from ai_framework.utils.agents.agent_config import RULES_ONLY

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
CREATE_PROVIDER = "ai_framework.ui.controllers.agents_controller.create_provider"


def sample_data():
    rng = np.random.default_rng(5)
    x = rng.normal(size=50)
    return pd.DataFrame(
        {"id": np.arange(50), "x": x, "noise": rng.normal(size=50),
         "y": 3 * x + 2 * rng.normal(size=50)}
    ).assign(leak=lambda d: d["y"] * 2 + 1)


def new_app():
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (sample_data(), "agents.csv", "agents-ui")
    app.run()
    app.sidebar.button(key="nav_eda_agent").click().run()
    assert not app.exception
    return app


def start(app, provider=RULES_ONLY):
    app.selectbox(key="agent_target").set_value("y")
    app.selectbox(key="agent_provider").set_value(provider)
    app.button[0].click()  # the form submit is the first button in the body
    app.run()
    assert not app.exception
    return app


def test_agent_is_under_exploration_and_navigating_does_not_analyze():
    app = new_app()
    assert any("Exploración / Agente IA" in m.value for m in app.markdown)
    assert "agent_eda_session" not in app.session_state


def test_view_is_translated_to_english():
    app = new_app()
    app.sidebar.selectbox(key="selector_idioma_global").set_value("en").run()
    assert not app.exception
    assert any("Exploration / AI Agent" in m.value for m in app.markdown)
    assert any(b.label == "Start agent" for b in app.button)


def test_without_llm_shows_findings_and_applies_suggestion():
    app = start(new_app())

    assert any("identificador" in warning.value for warning in app.warning)
    session = app.session_state["agent_eda_session"]
    assert session["agent"].llm is None
    assert "api_key" not in session["options"]
    assert session["blackboard"].source == "agents.csv"  # no local path towards the LLM

    app.button(key="agent_apply").click().run()
    assert app.session_state["vista_activa"] == "configuracion"
    assert app.session_state["sup_tarea"] == "regresion"
    assert app.session_state["sup_target"] == "y"
    assert app.session_state["sup_features"] == ["x", "noise"]


def test_fake_llm_interprets_and_answers_questions():
    llm = FakeLLMProvider([LLMResponse(text="Initial interpretation"), LLMResponse(text="No nulls.")])
    with patch(CREATE_PROVIDER, return_value=llm):
        app = start(new_app(), provider="gemini")
        assert "Initial interpretation" in [m.markdown[0].value for m in app.chat_message]

        app.chat_input(key="agent_eda_question").set_value("Any nulls?").run()
    texts = [m.markdown[0].value for m in app.chat_message]
    assert texts[-2:] == ["Any nulls?", "No nulls."]


def test_provider_error_is_shown():
    with patch(CREATE_PROVIDER, side_effect=ValueError("Defina GEMINI_API_KEY")):
        app = start(new_app(), provider="gemini")

    assert any("GEMINI_API_KEY" in error.value for error in app.error)
    assert "agent_eda_session" not in app.session_state


def test_changing_dataset_discards_the_agent_session():
    app = start(new_app())
    assert "agent_eda_session" in app.session_state

    app.session_state["fuente_aplicada"] = (sample_data().iloc[:30], "other.csv", "other")
    app.run()
    assert "agent_eda_session" not in app.session_state
