"""Agents answer in the interface language and report progress while they work."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_framework.agents import (
    CLUSTERING_REQUESTED,
    DATASET_LOADED,
    Blackboard,
    ClusteringAgent,
    EDAAgent,
    Event,
    EventBus,
    FakeLLMProvider,
    LLMResponse,
    RegressionAgent,
    ToolCall,
)
from ai_framework.agents.messages import MESSAGES, language_instruction, message
from ai_framework.utils.agents.agent_config import RULES_ONLY
from scripts.ejecutar_agentes import main

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
CREATE_PROVIDER = "ai_framework.ui.controllers.agents_controller.create_provider"


def eda_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=60)
    return pd.DataFrame({"id": np.arange(60), "x": x, "y": 2 * x + rng.normal(size=60)})


def run_eda(language, llm=None):
    updates = []
    blackboard = Blackboard(eda_data(), target="y")
    bus = EventBus()
    agent = EDAAgent(blackboard, llm, language=language, on_progress=updates.append).connect(bus)
    bus.emit(Event(DATASET_LOADED))
    return blackboard, agent, updates


# ---------------------------------------------------------------- catalog
def test_every_message_exists_in_every_language():
    assert set(MESSAGES["es"]) == set(MESSAGES["en"])


def test_unknown_language_falls_back_to_spanish():
    assert message("fr", "eda_no_target") == message("es", "eda_no_target")


# ---------------------------------------------------------------- language
@pytest.mark.parametrize(
    "language,expected",
    [("es", "parece un identificador"), ("en", "looks like an identifier")],
)
def test_findings_follow_the_language(language, expected):
    blackboard, _, _ = run_eda(language)

    assert any(expected in f.message for f in blackboard.findings)


def test_system_prompt_asks_for_the_users_language():
    llm = FakeLLMProvider([LLMResponse(text="Done.")])
    _, agent, _ = run_eda("en", llm)

    system = llm.calls[0]["messages"][0]["content"]
    assert agent.role in system
    assert language_instruction("en") in system and "English" in system
    assert "language the user wrote in" in system


def test_language_can_change_between_questions():
    llm = FakeLLMProvider([LLMResponse(text="Hecho."), LLMResponse(text="Done.")])
    _, agent, _ = run_eda("es", llm)
    agent.language = "en"
    agent.ask("Any nulls?")

    assert "Spanish" in llm.calls[0]["messages"][0]["content"]
    assert "English" in llm.calls[1]["messages"][0]["content"]


def test_fallback_texts_follow_the_language():
    blackboard, _, _ = run_eda("en", FakeLLMProvider([]))

    assert any(f.message.startswith("LLM unavailable") for f in blackboard.findings)


# ---------------------------------------------------------------- progress
def test_progress_is_reported_step_by_step_and_kept_on_the_blackboard():
    blackboard, _, updates = run_eda("en")

    assert updates[0] == "EDA agent: starting."
    assert updates[1].startswith("Profile ready: 60 rows")
    assert updates[2].startswith("Rules applied:")
    assert "Notifying other agents: eda_done." in updates
    assert updates[-1] == "EDA agent: done."
    assert [u for _, u in blackboard.progress] == updates


def test_progress_includes_the_llm_and_its_tools():
    llm = FakeLLMProvider([
        LLMResponse(tool_calls=(ToolCall("c1", "column_frequencies", {"column": "missing"}),)),
        LLMResponse(text="Done."),
    ])
    _, _, updates = run_eda("en", llm)

    assert "Asking the language model (step 1)…" in updates
    assert "The model is using the tool “column_frequencies”." in updates
    assert any("returned an error" in u for u in updates)
    assert "The model finished its interpretation." in updates


def test_progress_reports_a_failed_llm():
    _, _, updates = run_eda("es", FakeLLMProvider([]))

    assert "El modelo de lenguaje no respondió; se muestran sólo las reglas." in updates


def test_clustering_and_supervised_agents_report_their_own_steps():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    updates = []
    blackboard = Blackboard(data)
    bus = EventBus()
    ClusteringAgent(blackboard, language="es", on_progress=updates.append).connect(bus)
    bus.emit(Event(CLUSTERING_REQUESTED))
    assert any(u.startswith("K-Means evaluado de k=2") for u in updates)
    assert any(u.startswith("Partición elegida:") for u in updates)

    updates.clear()
    data["y"] = 2 * data["a"] + rng.normal(scale=0.2, size=80)
    blackboard = Blackboard(data, target="y")
    bus = EventBus()
    RegressionAgent(blackboard, language="en", on_progress=updates.append).connect(bus)
    bus.emit(Event("regression_requested"))
    assert any(u.startswith("Comparing 14 candidates") for u in updates)
    assert any(u.startswith("Comparison ready:") and "RMSE" in u for u in updates)


# ---------------------------------------------------------------- interface
def new_app(language):
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (eda_data(), "lang.csv", f"lang-{language}")
    app.run()
    if language != "es":
        app.sidebar.selectbox(key="selector_idioma_global").set_value(language).run()
    app.sidebar.button(key="nav_eda_agent").click().run()
    assert not app.exception
    return app


def start(app, provider=RULES_ONLY):
    app.selectbox(key="agent_target").set_value("y")
    app.selectbox(key="agent_provider").set_value(provider)
    app.button[0].click()
    app.run()
    assert not app.exception
    return app


def test_ui_agent_speaks_the_interface_language_and_keeps_the_activity_log():
    app = start(new_app("en"))

    assert any("looks like an identifier" in w.value for w in app.warning)
    assert app.session_state["agent_eda_session"]["agent"].language == "en"
    assert app.session_state["agent_eda_session"]["agent"].on_progress is None
    activity = next(e for e in app.expander if e.label == "Agent activity")
    assert "EDA agent: done." in activity.markdown[0].value
    assert any(s.label == "The agent finished." for s in app.status)


def test_ui_spanish_by_default():
    app = start(new_app("es"))

    assert any("parece un identificador" in w.value for w in app.warning)
    assert any(e.label == "Actividad del agente" for e in app.expander)


def test_ui_chat_uses_the_current_language():
    llm = FakeLLMProvider([LLMResponse(text="Hola."), LLMResponse(text="Hello.")])
    with patch(CREATE_PROVIDER, return_value=llm):
        app = start(new_app("es"), provider="gemini")
        app.sidebar.selectbox(key="selector_idioma_global").set_value("en").run()
        app.chat_input(key="agent_eda_question").set_value("Any nulls?").run()

    assert "Spanish" in llm.calls[0]["messages"][0]["content"]
    assert "English" in llm.calls[1]["messages"][0]["content"]


# ---------------------------------------------------------------- CLI
def test_cli_prints_progress_in_the_chosen_language(tmp_path, capsys):
    path = tmp_path / "data.csv"
    eda_data().to_csv(path, index=False)

    assert main([str(path), "--target", "y", "--language", "en"]) == 0
    output = capsys.readouterr().out
    assert "  - EDA agent: starting." in output
    assert "looks like an identifier" in output

    assert main([str(path), "--target", "y", "--idioma", "es"]) == 0
    assert "  - Agente EDA: iniciando." in capsys.readouterr().out
