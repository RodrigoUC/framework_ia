"""Agents: bus, blackboard, tools, loop with a fake LLM, EDAAgent and CLI."""

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_framework.agents import (
    DATASET_LOADED,
    EDA_DONE,
    Agent,
    AgenteEDA,
    Blackboard,
    EDAAgent,
    Event,
    EventBus,
    FakeLLMProvider,
    LLMResponse,
    OpenAICompatibleProvider,
    Pizarra,
    ProviderError,
    ToolCall,
    create_provider,
)
from ai_framework.agents.llm.provider import INVALID_JSON_KEY
from ai_framework.agents.tools.eda_tools import task_type
from ai_framework.agents.tools.registry import run_tool, tool
from ai_framework.data.eda import EDA
from ai_framework.utils.results import Finding, Hallazgo, ModelingSuggestion
from ai_framework.ui.models.agents_state import clear_agent_results, sync_eda_agent_context
from ai_framework.utils.agents.agent_config import RULES_ONLY, EDAThresholds
from scripts.ejecutar_agentes import main


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(7)
    x, noise = rng.normal(size=(2, 60))
    return pd.DataFrame(
        {
            "id": np.arange(60),
            "x": x,
            "noise": noise,
            "constant": 1.0,
            "missing": np.where(np.arange(60) % 2 == 0, np.nan, noise),
            "leak": 2 * x + 1,
            "y": 2 * x + 1 + 0.001 * noise,
        }
    )


@pytest.fixture
def imbalanced_data():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 100))
    return pd.DataFrame({"a": a, "b": b, "label": [1] * 5 + [0] * 95})


def build_system(data, target, llm=None):
    blackboard = Blackboard(data, target=target)
    bus = EventBus()
    agent = EDAAgent(blackboard, llm).connect(bus)
    received = []
    bus.subscribe(EDA_DONE, received.append)
    return blackboard, bus, agent, received


def tool_call(name, id_="c1", **arguments):
    return LLMResponse(tool_calls=(ToolCall(id_, name, arguments),))


def fake_client(message=None, error=None):
    requests = []

    def create(**parameters):
        requests.append(parameters)
        if error is not None:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), requests


# ---------------------------------------------------------------- infrastructure
def test_bus_processes_nested_events_in_fifo_order():
    bus = EventBus()
    order = []

    def first(event):
        order.append("a")
        bus.emit(Event("b"))
        order.append("a-end")

    bus.subscribe("a", first)
    bus.subscribe("b", lambda event: order.append("b"))
    bus.emit(Event("a"))

    assert order == ["a", "a-end", "b"]
    assert [e.type for e in bus.history] == ["a", "b"]


def test_blackboard_is_an_eda_with_shared_state(imbalanced_data):
    blackboard = Blackboard(imbalanced_data, target="label")

    assert isinstance(blackboard, EDA)
    assert blackboard.resumen_calidad()["filas"] == 100
    with pytest.raises(KeyError):
        Blackboard(imbalanced_data, target="z")


def test_blackboard_does_not_duplicate_findings_or_send_rows_to_the_llm():
    blackboard = Blackboard(pd.DataFrame({"secret": ["private-value-123"]}))
    finding = Finding("eda", "info", "x")

    assert blackboard.add_finding(finding) and not blackboard.add_finding(finding)
    context = blackboard.llm_context()
    assert "secret" in context and "private-value-123" not in context


def test_agent_is_abstract(imbalanced_data):
    with pytest.raises(TypeError):
        Agent(Blackboard(imbalanced_data))


def test_spanish_aliases_still_work():
    assert AgenteEDA is EDAAgent and Pizarra is Blackboard and Hallazgo is Finding


# ---------------------------------------------------------------- tools
@pytest.fixture
def demo():
    @tool("Demo.", n="Amount.")
    def demo(blackboard, name: str, n: int = 3, columns: list[str] | None = None) -> dict:
        if n < 0:
            raise ValueError("negative n")
        return {"name": name, "n": n}

    return demo


def test_schema_excludes_blackboard_and_marks_required(demo):
    parameters = demo.schema()["function"]["parameters"]

    assert set(parameters["properties"]) == {"name", "n", "columns"}
    assert parameters["required"] == ["name"]
    assert parameters["properties"]["n"] == {"type": "integer", "description": "Amount."}
    assert parameters["properties"]["columns"] == {"type": "array", "items": {"type": "string"}}


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("other", {}),
        ("demo", {"name": "a", "x": 1}),
        ("demo", {INVALID_JSON_KEY: "{"}),
        ("demo", {}),
        ("demo", {"name": "a", "n": -1}),
    ],
    ids=["unknown", "extra-argument", "invalid-json", "missing-required", "exception"],
)
def test_run_tool_returns_errors_instead_of_raising(demo, name, arguments):
    assert "error" in run_tool(ToolCall("1", name, arguments), [demo], None)


def test_run_tool_calls_the_tool(demo):
    assert run_tool(ToolCall("1", "demo", {"name": "a"}), [demo], None) == {"name": "a", "n": 3}


@pytest.mark.parametrize(
    "series,task",
    [(pd.Series(range(50)), "regresion"), (pd.Series([0, 1] * 25), "clasificacion"),
     (pd.Series(["a", "b"] * 25), "clasificacion")],
)
def test_task_type(series, task):
    assert task_type(series) == task


# ---------------------------------------------------------------- EDAAgent without LLM
def test_proactive_rules_detect_risks_and_enable_regression(regression_data):
    blackboard, bus, agent, received = build_system(regression_data, "y")
    bus.emit(Event(DATASET_LOADED))

    rules = {f.detail.get("column"): f.detail.get("rule") for f in blackboard.findings}
    assert rules == {
        "id": "identifier", "constant": "constant", "missing": "nulls",
        "x": "leakage", "leak": "leakage",
    }
    assert received[0].data == {"target": "y", "task": "regresion", "features": ["noise", "missing"]}
    assert received[0].source == "eda"
    assert agent.suggest_configuration() == ModelingSuggestion(
        "regresion", "y", ("noise", "missing"), ("constant", "id", "leak", "x")
    )


def test_thresholds_are_configurable(regression_data):
    blackboard = Blackboard(regression_data, target="y")
    EDAAgent(blackboard, thresholds=EDAThresholds(nulls=60.0)).connect(bus := EventBus())
    bus.emit(Event(DATASET_LOADED))

    assert "nulls" not in {f.detail.get("rule") for f in blackboard.findings}


def test_detects_imbalance_in_classification(imbalanced_data):
    blackboard, bus, _, received = build_system(imbalanced_data, "label")
    bus.emit(Event(DATASET_LOADED))

    assert received[0].data["task"] == "clasificacion"
    assert any(f.detail.get("rule") == "imbalance" for f in blackboard.findings)


def test_without_target_eda_done_is_not_emitted(imbalanced_data):
    blackboard, bus, agent, received = build_system(imbalanced_data, None)
    bus.emit(Event(DATASET_LOADED))

    assert received == []
    assert agent.suggest_configuration() is None
    assert any("objetivo" in f.message for f in blackboard.findings)


def test_ask_without_llm_fails_explicitly(imbalanced_data):
    _, _, agent, _ = build_system(imbalanced_data, "label")
    with pytest.raises(RuntimeError):
        agent.ask("What do you see?")


# ---------------------------------------------------------------- EDAAgent with a fake LLM
def test_loop_runs_tool_and_returns_interpretation(regression_data):
    llm = FakeLLMProvider([tool_call("correlations", threshold=0.9), LLMResponse(text="There is leakage.")])
    blackboard, bus, _, received = build_system(regression_data, "y", llm)
    bus.emit(Event(DATASET_LOADED))

    assert blackboard.messages[-1] == ("eda", "There is leakage.")
    assert len(received) == 1
    first, second = llm.calls
    assert {t["function"]["name"] for t in first["tools"]} == {
        "dataset_profile", "correlations", "outliers", "column_frequencies",
    }
    assert '"profile"' in first["messages"][1]["content"]
    assistant, tool_reply = second["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "correlations"
    assert (tool_reply["role"], tool_reply["tool_call_id"]) == ("tool", "c1")
    assert {"a": "leak", "b": "y", "r": 1.0} in json.loads(tool_reply["content"])["pairs"]


def test_tool_error_goes_back_to_the_llm(imbalanced_data):
    llm = FakeLLMProvider([tool_call("column_frequencies", column="missing"), LLMResponse(text="ok")])
    _, bus, _, _ = build_system(imbalanced_data, "label", llm)
    bus.emit(Event(DATASET_LOADED))

    assert "KeyError" in json.loads(llm.calls[1]["messages"][-1]["content"])["error"]


def test_provider_failure_does_not_stop_the_flow(imbalanced_data):
    blackboard, bus, _, received = build_system(imbalanced_data, "label", FakeLLMProvider([]))
    bus.emit(Event(DATASET_LOADED))

    assert any("LLM no disponible" in f.message for f in blackboard.findings)
    assert len(received) == 1
    assert "desbalanceado" in blackboard.messages[-1][1]


def test_step_limit(imbalanced_data):
    llm = FakeLLMProvider([tool_call("dataset_profile", id_=str(i)) for i in range(10)])
    _, _, agent, _ = build_system(imbalanced_data, "label", llm)
    agent.max_steps = 2

    assert "límite" in agent.ask("Analyze")
    assert len(llm.calls) == 2


def test_ask_records_the_conversation(imbalanced_data):
    llm = FakeLLMProvider([LLMResponse(text="Answer")])
    blackboard, _, agent, _ = build_system(imbalanced_data, "label", llm)

    assert agent.ask("Any nulls?") == "Answer"
    assert blackboard.messages == [("user", "Any nulls?"), ("eda", "Answer")]
    assert llm.calls[0]["messages"][1]["content"].startswith("Any nulls?")


def test_console_shows_prompt_tools_and_answer(capsys, imbalanced_data):
    llm = FakeLLMProvider([tool_call("correlations", threshold=0.9), LLMResponse(text="Final answer")])
    _, _, agent, _ = build_system(imbalanced_data, "label", llm)
    agent.ask("Any leakage?")

    output = capsys.readouterr().out
    assert agent.role in output
    assert 'correlations({"threshold": 0.9})' in output
    assert '"pairs"' in output
    assert "Final answer" in output
    # The initial prompt is not repeated even though the history is resent each step.
    assert output.count("Any leakage?") == 1


def test_console_shows_the_provider_error(capsys, imbalanced_data):
    _, _, agent, _ = build_system(imbalanced_data, "label", FakeLLMProvider([]))
    with pytest.raises(RuntimeError):
        agent.ask("Hello")

    assert "ERROR step 1" in capsys.readouterr().out


def test_console_without_character_support_does_not_stop_the_agent(monkeypatch, imbalanced_data):
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    llm = FakeLLMProvider([LLMResponse(text="Done \U0001F600")])
    _, _, agent, _ = build_system(imbalanced_data, "label", llm)

    assert agent.ask("Hello") == "Done \U0001F600"


# ---------------------------------------------------------------- provider
def test_provider_normalizes_tool_calls_and_invalid_json():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(id="a", function=SimpleNamespace(name="f", arguments='{"x": 1}')),
            SimpleNamespace(id="b", function=SimpleNamespace(name="g", arguments="{broken")),
        ],
    )
    client, requests = fake_client(message)
    response = OpenAICompatibleProvider("groq", "model-x", client=client).complete(
        [{"role": "user", "content": "hi"}], [{"type": "function"}]
    )

    assert response.tool_calls[0] == ToolCall("a", "f", {"x": 1})
    assert response.tool_calls[1].arguments == {INVALID_JSON_KEY: "{broken"}
    assert requests[0]["model"] == "model-x" and "tools" in requests[0]


def test_thought_signature_goes_back_to_gemini_with_the_real_sdk(imbalanced_data):
    """Runs the real openai SDK against a server on 127.0.0.1, without leaving the machine."""
    openai = pytest.importorskip("openai")
    signature = {"google": {"thought_signature": "opaque-signature"}}
    requests = []

    class Server(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            message = {"role": "assistant", "content": "Done"}
            if len(requests) == 1:
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "c1", "type": "function", "extra_content": signature,
                        "function": {"name": "correlations", "arguments": '{"threshold": 0.5}'},
                    }],
                }
            body = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = openai.OpenAI(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="k", max_retries=0
        )
        llm = OpenAICompatibleProvider("gemini", "m", client=client)
        _, _, agent, _ = build_system(imbalanced_data, "label", llm)

        assert agent.ask("Analyze") == "Done"
    finally:
        server.shutdown()
        server.server_close()
    assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["extra_content"] == signature


def test_provider_without_tools_does_not_send_tools():
    client, requests = fake_client(SimpleNamespace(content="hi", tool_calls=None))

    assert OpenAICompatibleProvider("ollama", "m", client=client).complete([], []) == LLMResponse(text="hi")
    assert "tools" not in requests[0]


def test_http_error_includes_reason_and_hint():
    class HTTPError(Exception):
        status_code = 403
        body = [{"error": {"message": "Generative Language API has not been used"}}]

    client, _ = fake_client(error=HTTPError("Error code: 403"))
    with pytest.raises(ProviderError) as context:
        OpenAICompatibleProvider("gemini", "m", client=client).complete([], [])

    assert context.value.status == 403
    assert "Generative Language API has not been used" in str(context.value)
    assert "habilite la API" in str(context.value)


def test_provider_validates_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("unknown", "m", client=object())
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("gemini", "m")


def test_default_model_is_the_suggested_one(monkeypatch):
    monkeypatch.delenv("AGENTES_MODELO", raising=False)
    assert OpenAICompatibleProvider("groq", client=object()).model == "llama-3.3-70b-versatile"


def test_create_provider_rules_only():
    assert create_provider(RULES_ONLY) is None


# ---------------------------------------------------------------- UI state
def test_agent_state_is_invalidated_when_data_changes(imbalanced_data):
    state = {"dataset_identidad": "a"}
    sync_eda_agent_context(state, imbalanced_data)
    state["agent_eda_session"] = object()

    sync_eda_agent_context(state, imbalanced_data)
    assert "agent_eda_session" in state
    sync_eda_agent_context(state, imbalanced_data.iloc[:50])
    assert "agent_eda_session" not in state

    state["agent_eda_session"] = object()
    clear_agent_results(state)
    assert not {"agent_eda_session", "agent_eda_context"} & set(state)


# ---------------------------------------------------------------- CLI
def test_cli_without_llm_prints_findings_and_suggestion(tmp_path, capsys, regression_data):
    path = tmp_path / "data.csv"
    regression_data.to_csv(path, index=False)

    assert main([str(path), "--target", "y"]) == 0
    output = capsys.readouterr().out
    assert "EDA done: task = regresion" in output
    assert "posible fuga" in output
    assert "Suggestion: regresion on y" in output


def test_cli_keeps_the_spanish_flags(tmp_path, capsys, regression_data):
    path = tmp_path / "data.csv"
    regression_data.to_csv(path, index=False)

    assert main([str(path), "--agente", "eda", "--proveedor", "reglas", "--target", "y"]) == 0
    assert "EDA done" in capsys.readouterr().out


def test_cli_interactive_requires_llm(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.csv"), "--interactive"])
