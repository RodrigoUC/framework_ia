"""ClusteringAgent: deterministic search, rules, tools, view and CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_framework.agents import (
    CLUSTERING_DONE,
    CLUSTERING_REQUESTED,
    Blackboard,
    ClusteringAgent,
    Event,
    EventBus,
    FakeLLMProvider,
    LLMResponse,
    ToolCall,
)
from ai_framework.agents.tools.clustering_tools import (
    create_clustering,
    evaluate_hierarchical,
    evaluate_k,
    run_clustering,
    summarize_result,
)
from ai_framework.agents.tools.registry import run_tool
from ai_framework.utils.results import ClusteringSuggestion
from ai_framework.ui.models.agents_state import (
    clear_agent_results,
    sync_clustering_agent_context,
    sync_eda_agent_context,
)
from ai_framework.utils.agents.agent_config import RULES_ONLY, ClusteringThresholds
from scripts.ejecutar_agentes import main

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
CREATE_PROVIDER = "ai_framework.ui.controllers.agents_controller.create_provider"


def blobs(rows_per_group=30, centers=((0, 0), (10, 10), (-10, 10))):
    rng = np.random.default_rng(11)
    rows = [rng.normal(center, 0.4, size=(rows_per_group, 2)) for center in centers]
    return pd.DataFrame(np.vstack(rows), columns=["a", "b"])


def build_system(data, llm=None, **options):
    blackboard = Blackboard(data)
    bus = EventBus()
    agent = ClusteringAgent(blackboard, llm, **options).connect(bus)
    received = []
    bus.subscribe(CLUSTERING_DONE, received.append)
    return blackboard, bus, agent, received


def rules(blackboard):
    return {f.detail.get("rule") for f in blackboard.findings}


def test_finds_the_real_groups_without_llm():
    blackboard, bus, agent, received = build_system(blobs())
    bus.emit(Event(CLUSTERING_REQUESTED))

    suggestion = agent.suggest_clustering()
    assert isinstance(suggestion, ClusteringSuggestion)
    assert suggestion.n_clusters == 3
    assert suggestion.silhouette > 0.8
    assert suggestion.sizes == (30, 30, 30)
    assert rules(blackboard) == {"recommendation"}
    assert received[0].data["n_clusters"] == 3
    assert received[0].source == "clustering"


def test_without_structure_warns_it_is_weak():
    rng = np.random.default_rng(1)
    blackboard, bus, _, _ = build_system(pd.DataFrame(rng.uniform(size=(150, 6))))
    bus.emit(Event(CLUSTERING_REQUESTED))

    assert "weak_structure" in rules(blackboard)


def test_warns_about_a_small_cluster():
    data = pd.concat([blobs(100, ((0, 0), (10, 10))), pd.DataFrame({"a": [60.0] * 3, "b": [60.0] * 3})])
    blackboard, bus, agent, _ = build_system(data.reset_index(drop=True))
    bus.emit(Event(CLUSTERING_REQUESTED))

    assert "small_cluster" in rules(blackboard)
    assert min(agent.suggest_clustering().sizes) == 3


def test_warns_about_variable_risks():
    data = blobs().assign(
        code=lambda d: np.arange(len(d)),
        constant=1.0,
        incomplete=lambda d: d["a"].where(d.index % 2 == 0),
    )
    blackboard, bus, _, _ = build_system(data, standardize=False)
    bus.emit(Event(CLUSTERING_REQUESTED))

    assert {"identifier", "dropped", "nulls", "unscaled"} <= rules(blackboard)


def test_thresholds_are_configurable():
    blackboard, bus, agent, _ = build_system(blobs(), thresholds=ClusteringThresholds(k_max=2))
    bus.emit(Event(CLUSTERING_REQUESTED))

    assert agent.suggest_clustering().n_clusters == 2
    assert blackboard.results["kmeans_evaluation"]["k"].max() == 2


def test_unencoded_categorical_columns_fail_before_calling_the_llm():
    llm = FakeLLMProvider([])
    data = blobs().assign(kind=lambda d: np.where(d.index % 2 == 0, "x", "y"))
    _, bus, _, _ = build_system(data, llm, features=["a", "kind"])

    with pytest.raises(TypeError):
        bus.emit(Event(CLUSTERING_REQUESTED))
    assert llm.calls == []


def test_llm_context_exposes_neither_rows_nor_tiny_groups():
    data = blobs(20, ((0, 0), (10, 10))).assign(a=lambda d: d["a"] + 0.123456789)
    data.loc[0, ["a", "b"]] = [987654.0, 987654.0]
    blackboard, bus, _, _ = build_system(data)
    bus.emit(Event(CLUSTERING_REQUESTED))

    context = blackboard.llm_context()
    assert "987654" not in context
    assert "0.123456789" not in context
    assert "cluster_result" not in context
    isolated = next(g for g in json.loads(context)["clustering"]["groups"] if g["size"] == 1)
    assert "distinctive" not in isolated


def test_llm_interprets_with_context_and_tools():
    llm = FakeLLMProvider([
        LLMResponse(tool_calls=(ToolCall("c1", "run_clustering", {
            "algorithm": "K-Medoids", "n_clusters": 3,
        }),)),
        LLMResponse(text="There are three groups."),
    ])
    blackboard, bus, _, _ = build_system(blobs(), llm)
    bus.emit(Event(CLUSTERING_REQUESTED))

    assert blackboard.messages[-1] == ("clustering", "There are three groups.")
    first, second = llm.calls
    assert '"clustering"' in first["messages"][1]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == {
        "evaluate_k", "evaluate_hierarchical", "run_clustering",
    }
    reply = json.loads(second["messages"][-1]["content"])
    assert reply["algorithm"] == "K-Medoids"
    # The LLM's exploration does not replace the deterministic recommendation.
    assert blackboard.results["clustering"]["algorithm"] != "K-Medoids"


def test_llm_failure_keeps_the_recommendation():
    blackboard, bus, agent, received = build_system(blobs(), FakeLLMProvider([]))
    bus.emit(Event(CLUSTERING_REQUESTED))

    assert any("LLM no disponible" in f.message for f in blackboard.findings)
    assert agent.suggest_clustering().n_clusters == 3
    assert len(received) == 1


def test_clustering_tools():
    blackboard = Blackboard(blobs())
    create_clustering(blackboard)

    assert [r["k"] for r in evaluate_k.function(blackboard, 2, 4)["rows"]] == [2, 3, 4]
    rows = evaluate_hierarchical.function(blackboard, 2, 3, ["ward", "single"])["rows"]
    assert {r["metodo"] for r in rows} == {"ward", "single"}
    hac = run_clustering.function(blackboard, "HAC", 3, "average")
    assert hac["algorithm"] == "HAC-average" and len(hac["groups"]) == 3
    assert set(hac["groups"][0]["distinctive"]) == {"a", "b"}


@pytest.mark.parametrize(
    "arguments",
    [{"algorithm": "ACP", "n_clusters": 2}, {"algorithm": "HAC", "n_clusters": 2, "hac_method": "other"},
     {"algorithm": "K-Means", "n_clusters": 100000}],
    ids=["not-clustering", "invalid-method", "excessive-k"],
)
def test_tool_returns_errors_to_the_llm(arguments):
    blackboard = Blackboard(blobs())
    create_clustering(blackboard)
    call = ToolCall("1", "run_clustering", arguments)

    assert "error" in run_tool(call, [run_clustering], blackboard)


def test_summary_measures_distinctive_variables_in_standard_deviations():
    model = create_clustering(Blackboard(blobs()))
    summary = summarize_result(model.K_means(3))

    assert sum(g["size"] for g in summary["groups"]) == 90
    assert all(abs(v) < 5 for g in summary["groups"] for v in g["distinctive"].values())


def test_agent_states_do_not_overwrite_each_other():
    data = blobs()
    config = {"features": ["a", "b"], "incluir_categoricas": False, "estandarizar": True}
    state = {"dataset_identidad": "x"}
    sync_clustering_agent_context(state, data, config)
    state["agent_clustering_session"] = object()

    # Opening the EDA agent view must not erase the clustering session.
    sync_eda_agent_context(state, data)
    assert "agent_clustering_session" in state
    sync_clustering_agent_context(state, data, config)
    assert "agent_clustering_session" in state
    sync_clustering_agent_context(state, data, {**config, "estandarizar": False})
    assert "agent_clustering_session" not in state

    state["agent_clustering_session"] = object()
    clear_agent_results(state)
    assert "agent_clustering_session" not in state


# ---------------------------------------------------------------- view
def new_app(data=None):
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (blobs() if data is None else data, "blobs.csv", "blobs-ui")
    app.run()
    app.sidebar.button(key="nav_clustering_agent").click().run()
    assert not app.exception
    return app


def start(app, provider=RULES_ONLY):
    app.selectbox(key="agent_provider").set_value(provider)
    next(b for b in app.button if b.label == "Iniciar agente").click()
    app.run()
    assert not app.exception
    return app


def test_navigating_does_not_cluster_until_the_button_is_pressed():
    app = new_app()

    assert "agent_clustering_session" not in app.session_state
    assert any("Iniciar agente" in i.value for i in app.info)


def test_view_shows_the_recommendation_without_llm():
    app = start(new_app())

    session = app.session_state["agent_clustering_session"]
    assert session["agent"].llm is None
    assert session["blackboard"].source == "blobs.csv"
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Clusters"] == "3"
    assert float(metrics["Silhouette"]) > 0.8


def test_changing_variables_invalidates_the_recommendation():
    app = start(new_app())
    assert "agent_clustering_session" in app.session_state

    app.multiselect(key="modelo_features").set_value(["a"]).run()
    assert not app.exception
    assert "agent_clustering_session" not in app.session_state


def test_view_with_fake_llm_answers_questions():
    llm = FakeLLMProvider([LLMResponse(text="Three groups."), LLMResponse(text="Group 0.")])
    with patch(CREATE_PROVIDER, return_value=llm):
        app = start(new_app(), provider="gemini")
        app.chat_input(key="agent_clustering_question").set_value("Which is larger?").run()

    texts = [m.markdown[0].value for m in app.chat_message]
    assert texts == ["Three groups.", "Which is larger?", "Group 0."]


def test_provider_error_in_the_view():
    with patch(CREATE_PROVIDER, side_effect=ValueError("Defina GEMINI_API_KEY")):
        app = start(new_app(), provider="gemini")

    assert any("GEMINI_API_KEY" in error.value for error in app.error)
    assert "agent_clustering_session" not in app.session_state


# ---------------------------------------------------------------- CLI
def test_cli_clustering(tmp_path, capsys):
    path = tmp_path / "blobs.csv"
    blobs().to_csv(path, index=False)

    assert main([str(path), "--agent", "clustering"]) == 0
    assert "k=3" in capsys.readouterr().out
