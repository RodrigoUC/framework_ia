"""Exercise the coordinated analysis through the Streamlit navigation."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_framework.agents import FakeLLMProvider, LLMResponse, ProviderError

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
DATA = ENTRYPOINT.parents[1] / "data"
CREATE_PROVIDER = "ai_framework.ui.controllers.agents_controller.create_provider"


def navigate(data: pd.DataFrame) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=120)
    app.session_state["fuente_aplicada"] = (data, "reference.csv", "reference")
    app.run()
    menu = [button.key for button in app.sidebar.button]
    index = menu.index("nav_coordinated_analysis")
    assert menu[index - 1:index + 2] == ["nav_configuracion", "nav_coordinated_analysis", "nav_clustering_agent"]
    app.sidebar.button(key="nav_coordinated_analysis").click().run()
    assert not app.exception
    assert "coordinated_result" not in app.session_state
    return app


def run_analysis(app: AppTest, *, target="target", task="auto") -> dict:
    app.selectbox(key="coordinated_target").set_value(target)
    app.selectbox(key="coordinated_task").set_value(task)
    app.button(key="coordinated_run").click().run()
    assert not app.exception
    assert app.get("download_button")
    return app.session_state["coordinated_result"]


@pytest.mark.parametrize(("filename", "expected"), [
    ("diabetes_lab03.csv", "regression"), ("breast_cancer.csv", "classification"),
])
def test_coordinator_routes_only_after_click(filename, expected):
    app = navigate(pd.read_csv(DATA / filename))
    app.sidebar.selectbox(key="selector_idioma_global").set_value("en").run()
    assert app.button(key="coordinated_run").label == "Run coordinated analysis"
    trace = run_analysis(app)
    assert trace["status"] == "completed"
    assert [event["type"] for event in trace["events"]].count(f"{expected}_requested") == 1
    assert any(metric.value == trace["result"]["algorithm"] for metric in app.metric)
    if expected == "regression":
        app.sidebar.button(key="nav_eda_agent").click().run()
        app.sidebar.button(key="nav_coordinated_analysis").click().run()
        assert app.session_state["coordinated_result"] == trace
        app.selectbox(key="coordinated_target").set_value(None).run()
        assert "coordinated_result" not in app.session_state


@pytest.mark.parametrize("eligible", [True, False])
def test_coordinator_shows_leakage_decisions(eligible):
    data = pd.read_csv(DATA / "diabetes_lab03.csv")
    if not eligible:
        data = data[["target"]]
    app = navigate(data.assign(target_copy=data["target"]))
    trace = json.loads(json.dumps(run_analysis(app)))
    assert any(d.get("column") == "target_copy" for d in trace["decisions"])
    assert any("target_copy" in str(item.value) for item in app.markdown)
    assert trace["status"] == ("completed" if eligible else "needs_review")
    if not eligible:
        assert not app.metric
        assert any("Se requiere revisión" in warning.value for warning in app.warning)


def test_coordinator_explores_without_target_and_invalidates_stale_results():
    data = pd.read_csv(DATA / "diabetes_lab03.csv")
    app = navigate(data)
    trace = run_analysis(app, target=None)
    assert trace["result"]["rows"] == 442
    assert [task["agent"] for task in trace["tasks"]] == ["eda"]
    app.selectbox(key="coordinated_task").set_value("clustering").run()
    assert "coordinated_result" not in app.session_state
    trace = run_analysis(app, task="clustering")
    assert trace["status"] == "completed"
    assignment = next(event for event in trace["events"] if event["type"] == "clustering_requested")
    assert "target" not in assignment["data"]["features"]
    app.session_state["fuente_aplicada"] = (data.iloc[:20], "new.csv", "new")
    app.run()
    assert "coordinated_result" not in app.session_state


def test_coordinator_explains_and_answers_without_repeating_analysis():
    app = navigate(pd.read_csv(DATA / "diabetes_lab03.csv"))
    app.selectbox(key="agent_provider").set_value("groq").run()
    app.text_input(key="agent_api_key").set_value("test-key").run()
    provider = FakeLLMProvider([LLMResponse(text="Regression completed."), LLMResponse(text="RMSE 53.282.")])
    with patch(CREATE_PROVIDER, return_value=provider) as create_provider:
        trace = run_analysis(app)
        assert trace["result"]["algorithm"] == "Lasso"
        assert not app.chat_input
        create_provider.assert_not_called()
        app.checkbox(key="coordinated_use_llm").check().run()
        assert "Regression completed." in app.chat_message[0].markdown[0].value
        app.chat_input(key="coordinated_question").set_value("What was the test error?").run()
        assert "RMSE 53.282." in app.chat_message[-1].markdown[0].value
        assert app.session_state["coordinated_result"] == trace
        assert [call["tools"] for call in provider.calls] == [[], []]
        assert create_provider.call_args.args[2] == "test-key"
        assert "test-key" not in json.dumps(provider.calls)
        assert "test-key" not in json.dumps(trace)
        assert provider.calls[-1]["messages"][-1]["content"] == "What was the test error?"
        initial = provider.calls[0]["messages"]
        assert [message["role"] for message in initial] == ["system", "user"]
        trace_json = initial[1]["content"].split("\n", 1)[1].split("\n\nQuestion:", 1)[0]
        assert json.loads(trace_json)["result"] == trace["result"]

        app.selectbox(key="agent_provider").set_value("rules").run()
        assert not app.chat_input
        assert len(provider.calls) == 2


def test_coordinator_keeps_trace_when_language_provider_fails():
    app = navigate(pd.read_csv(DATA / "diabetes_lab03.csv"))
    app.selectbox(key="agent_provider").set_value("groq").run()
    app.checkbox(key="coordinated_use_llm").check().run()
    with patch(CREATE_PROVIDER, side_effect=ValueError("Private credential detail")):
        trace = run_analysis(app, target=None)
    assert trace["status"] == "completed"
    assert app.chat_input
    assert any("no respondió" in warning.value for warning in app.warning)
    assert all("Private credential detail" not in warning.value for warning in app.warning)


def test_coordinator_reports_provider_status_and_can_retry_without_training():
    app = navigate(pd.read_csv(DATA / "diabetes_lab03.csv"))
    app.selectbox(key="agent_provider").set_value("groq").run()
    app.checkbox(key="coordinated_use_llm").check().run()
    provider = FakeLLMProvider([LLMResponse(text="EDA completed.")])
    error = ProviderError("groq", "openai/gpt-oss-120b", 429, "Private diagnostic")
    with patch(CREATE_PROVIDER, side_effect=[error, provider]):
        trace = run_analysis(app, target=None)
        assert any("HTTP 429" in warning.value for warning in app.warning)
        assert all("Private diagnostic" not in warning.value for warning in app.warning)
        app.button(key="coordinated_llm_retry").click().run()
    assert app.session_state["coordinated_result"] == trace
    assert any("EDA completed." in message.markdown[0].value for message in app.chat_message)
