"""RegressionAgent and ClassificationAgent: benchmark, rules, tools, chaining, view and CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_framework.agents import (
    CLASSIFICATION_DONE,
    CLASSIFICATION_REQUESTED,
    DATASET_LOADED,
    EDA_DONE,
    REGRESSION_DONE,
    REGRESSION_REQUESTED,
    AgenteRegresion,
    Blackboard,
    ClassificationAgent,
    EDAAgent,
    Event,
    EventBus,
    FakeLLMProvider,
    LLMResponse,
    RegressionAgent,
    ToolCall,
)
from ai_framework.agents.tools.classification_tools import (
    classification_feature_importance,
    compare_classifiers,
    confusion_matrix,
    train_classifier,
)
from ai_framework.agents.tools.regression_tools import (
    compare_regressors,
    regression_feature_importance,
    residual_summary,
    train_regressor,
)
from ai_framework.agents.tools.registry import run_tool
from ai_framework.ui.models.agents_state import clear_agent_results, sync_supervised_agent_context
from ai_framework.utils.agents.agent_config import RULES_ONLY, SupervisedThresholds
from ai_framework.utils.results import ModelRecommendation
from scripts.ejecutar_agentes import main

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
CREATE_PROVIDER = "ai_framework.ui.controllers.agents_controller.create_provider"


def linear_data(rows=120, noise=0.3):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(rows, 3))
    data = pd.DataFrame(x, columns=["a", "b", "c"])
    data["y"] = 1.5 * x[:, 0] - x[:, 1] + rng.normal(scale=noise, size=rows)
    return data


def binary_data(rows=120):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(rows, 3))
    data = pd.DataFrame(x, columns=["a", "b", "c"])
    data["label"] = np.where(x[:, 0] + 0.5 * x[:, 1] > 0, "yes", "no")
    return data


def build(agent_class, data, target, llm=None, **options):
    blackboard = Blackboard(data, target=target)
    bus = EventBus()
    agent = agent_class(blackboard, llm, **options).connect(bus)
    received = []
    bus.subscribe(REGRESSION_DONE, received.append)
    bus.subscribe(CLASSIFICATION_DONE, received.append)
    return blackboard, bus, agent, received


def rules(blackboard):
    return {f.detail.get("rule") for f in blackboard.findings}


# ---------------------------------------------------------------- regression
@pytest.fixture(scope="module")
def regression_run():
    blackboard, bus, agent, received = build(RegressionAgent, linear_data(), "y")
    bus.emit(Event(REGRESSION_REQUESTED))
    return blackboard, agent, received


def test_regression_recommends_a_good_model(regression_run):
    blackboard, agent, received = regression_run
    recommendation = agent.suggest_model()

    assert isinstance(recommendation, ModelRecommendation)
    assert recommendation.task == "regresion" and recommendation.metric == "rmse"
    assert recommendation.test_metrics["r2"] > 0.9
    assert rules(blackboard) == {"recommendation"}
    assert received[0].source == "regression"
    assert received[0].data["algorithm"] == recommendation.algorithm


def test_regression_context_reaches_the_llm_without_rows(regression_run):
    blackboard, _, _ = regression_run
    context = json.loads(blackboard.llm_context())

    assert context["regression"]["selection_metric"] == "rmse"
    assert "regression_experiment" not in context
    assert str(round(blackboard.datos["y"].iloc[0], 6)) not in blackboard.llm_context()


def test_regression_tools(regression_run):
    blackboard, agent, _ = regression_run

    ranking = compare_regressors.function(blackboard)["rows"]
    assert ranking[0]["validation_rmse"] <= ranking[-1]["validation_rmse"] or ranking[-1]["validation_rmse"] is None
    assert any(row["selected"] for row in ranking)
    single = train_regressor.function(blackboard, "RLM")
    assert single["test"]["r2"] > 0.9 and "rmse" in single["train"]
    assert train_regressor.function(blackboard, "RLS")["algorithm"] == "RLS"
    residuals = residual_summary.function(blackboard)
    assert abs(residuals["mean"]) < 0.3
    importance = regression_feature_importance.function(blackboard)
    if "features" in importance:
        assert importance["features"][0]["feature"] in {"a", "b"}
    assert "error" in run_tool(ToolCall("1", "train_regressor", {"algorithm": "XYZ"}), agent.tools, blackboard)


def test_regression_without_signal_is_flagged():
    rng = np.random.default_rng(9)
    data = pd.DataFrame(rng.normal(size=(120, 4)), columns=["a", "b", "c", "y"])
    blackboard, bus, _, _ = build(RegressionAgent, data, "y")
    bus.emit(Event(REGRESSION_REQUESTED))

    assert rules(blackboard) & {"no_better_than_mean", "weak_model"}


def test_regression_warns_about_few_rows_per_feature():
    blackboard, bus, _, _ = build(
        RegressionAgent, linear_data(), "y",
        thresholds=SupervisedThresholds(min_rows_per_feature=1000),
    )
    bus.emit(Event(REGRESSION_REQUESTED))

    assert "few_rows" in rules(blackboard)


def test_supervised_agent_requires_a_target():
    _, bus, _, _ = build(RegressionAgent, linear_data(), None)
    with pytest.raises(ValueError):
        bus.emit(Event(REGRESSION_REQUESTED))


def test_spanish_alias():
    assert AgenteRegresion is RegressionAgent


# ---------------------------------------------------------------- classification
@pytest.fixture(scope="module")
def classification_run():
    blackboard, bus, agent, received = build(ClassificationAgent, binary_data(), "label")
    bus.emit(Event(CLASSIFICATION_REQUESTED))
    return blackboard, agent, received


def test_classification_recommends_a_good_model(classification_run):
    blackboard, agent, received = classification_run
    recommendation = agent.suggest_model()

    assert recommendation.task == "clasificacion" and recommendation.metric == "f1_macro"
    assert recommendation.test_metrics["f1_macro"] > 0.8
    assert "recommendation" in rules(blackboard)
    assert "no_better_than_majority" not in rules(blackboard)
    assert received[0].source == "classification"


def test_classification_tools(classification_run):
    blackboard, agent, _ = classification_run

    ranking = compare_classifiers.function(blackboard)["rows"]
    assert {row["algorithm"] for row in ranking} >= {"KNN", "RF", "RL"}
    assert train_classifier.function(blackboard, "DT")["test"]["accuracy"] > 0.6
    importance = classification_feature_importance.function(blackboard)["features"]
    assert importance[0]["feature"] in {"a", "b"}
    matrix = confusion_matrix.function(blackboard)
    assert sorted(matrix["labels"]) == ["no", "yes"]
    assert sum(map(sum, matrix["matrix"])) == blackboard.results["classification"]["test_rows"]


def test_classification_flags_a_model_no_better_than_the_majority():
    rng = np.random.default_rng(5)
    data = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    data["label"] = ["rare"] * 15 + ["common"] * 135
    blackboard, bus, _, _ = build(ClassificationAgent, data, "label")
    bus.emit(Event(CLASSIFICATION_REQUESTED))

    assert rules(blackboard) & {"no_better_than_majority", "weak_class", "misleading_accuracy"}


def test_classification_with_fake_llm_uses_its_tools():
    llm = FakeLLMProvider([
        LLMResponse(tool_calls=(ToolCall("c1", "confusion_matrix", {}),)),
        LLMResponse(text="RL wins."),
    ])
    blackboard, bus, _, _ = build(ClassificationAgent, binary_data(), "label", llm)
    bus.emit(Event(CLASSIFICATION_REQUESTED))

    assert blackboard.messages[-1] == ("classification", "RL wins.")
    first, second = llm.calls
    assert '"classification"' in first["messages"][1]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == {
        "compare_classifiers", "train_classifier", "classification_feature_importance", "confusion_matrix",
    }
    assert "matrix" in json.loads(second["messages"][-1]["content"])


# ---------------------------------------------------------------- chaining
def test_eda_hands_off_to_the_matching_supervised_agent():
    data = linear_data().assign(id=lambda d: np.arange(len(d)))
    blackboard = Blackboard(data, target="y")
    bus = EventBus()
    EDAAgent(blackboard).connect(bus)
    regression = RegressionAgent(blackboard).connect(bus)
    classification = ClassificationAgent(blackboard).connect(bus)
    bus.emit(Event(DATASET_LOADED))

    assert [e.type for e in bus.history] == [DATASET_LOADED, EDA_DONE, REGRESSION_DONE]
    assert regression.suggest_model() is not None
    assert classification.suggest_model() is None
    # The identifier flagged by the EDA agent is not used to train.
    assert "id" not in blackboard.results["supervised_features"]


# ---------------------------------------------------------------- UI state
def test_supervised_state_is_invalidated_by_settings():
    data = linear_data()
    config = {"target": "y", "features": ["a", "b"], "test_size": 0.25, "particion_global": None}
    state = {"dataset_identidad": "x"}
    sync_supervised_agent_context(state, data, config, "regression")
    state["agent_regression_session"] = object()

    sync_supervised_agent_context(state, data, config, "regression")
    assert "agent_regression_session" in state
    sync_supervised_agent_context(state, data, {**config, "features": ["a"]}, "regression")
    assert "agent_regression_session" not in state

    state["agent_classification_session"] = object()
    clear_agent_results(state)
    assert "agent_classification_session" not in state


# ---------------------------------------------------------------- views
def new_app(data, task):
    app = AppTest.from_file(ENTRYPOINT, default_timeout=120)
    app.session_state["fuente_aplicada"] = (data, "supervised.csv", f"supervised-{task}")
    app.run()
    app.sidebar.button(key="nav_configuracion").click().run()
    app.radio(key="sup_tarea").set_value(task).run()
    assert not app.exception
    return app


def start(app, provider=RULES_ONLY):
    app.selectbox(key="agent_provider").set_value(provider)
    next(b for b in app.button if b.label == "Iniciar agente").click()
    app.run()
    assert not app.exception
    return app


def test_regression_view_trains_only_on_demand_and_shows_the_model():
    app = new_app(linear_data(), "regresion")
    app.sidebar.button(key="nav_regression_agent").click().run()
    assert not app.exception
    assert "agent_regression_session" not in app.session_state

    start(app)
    session = app.session_state["agent_regression_session"]
    assert session["agent"].llm is None
    assert session["blackboard"].source == "supervised.csv"
    metrics = {m.label: m.value for m in app.metric}
    assert float(metrics["R² prueba"]) > 0.9


def test_classification_view_with_fake_llm():
    llm = FakeLLMProvider([LLMResponse(text="Good classifier."), LLMResponse(text="Class yes.")])
    app = new_app(binary_data(), "clasificacion")
    app.sidebar.button(key="nav_classification_agent").click().run()
    with patch(CREATE_PROVIDER, return_value=llm):
        start(app, provider="gemini")
        app.chat_input(key="agent_classification_question").set_value("Which class fails?").run()

    texts = [m.markdown[0].value for m in app.chat_message]
    assert texts == ["Good classifier.", "Which class fails?", "Class yes."]
    assert any(m.label == "F1 macro prueba" for m in app.metric)


# ---------------------------------------------------------------- CLI
def test_cli_regression_and_pipeline(tmp_path, capsys):
    path = tmp_path / "linear.csv"
    linear_data().to_csv(path, index=False)

    assert main([str(path), "--agent", "regression", "--target", "y"]) == 0
    assert "Recommended model:" in capsys.readouterr().out

    assert main([str(path), "--agent", "pipeline", "--target", "y"]) == 0
    output = capsys.readouterr().out
    assert "EDA done: task = regresion" in output
    assert "[regression]" in output
