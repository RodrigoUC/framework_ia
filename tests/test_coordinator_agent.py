"""Verify the coordinated CLI routes requests through the event bus once."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ai_framework.agents.regression_agent import RegressionAgent
from scripts.run_agents import create_parser, main, run

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize(
    ("filename", "task", "target", "specialist", "expected_events"),
    [
        ("diabetes_lab03.csv", "auto", "target", "regression",
         ["eda_done", "regression_requested", "regression_done"]),
        ("breast_cancer.csv", "auto", "target", "classification",
         ["eda_done", "classification_requested", "classification_done"]),
        ("diabetes_lab03.csv", "clustering", "target", "clustering",
         ["eda_done", "clustering_requested", "clustering_done"]),
        ("diabetes_lab03.csv", "auto", None, "eda", []),
    ],
)
def test_coordinated_cli_routes_once(filename, task, target, specialist, expected_events, capsys):
    arguments = [str(DATA / filename), "--agent", "coordinated", "--task", task, "--language", "en"]
    if target is not None:
        arguments.extend(["--target", target])

    agent = run(create_parser().parse_args(arguments))

    assert agent.name == specialist
    assert [event.type for event in agent.bus.history] == ["dataset_loaded", *expected_events]
    assert agent.bus.history[0].source == "coordinator"
    if specialist != "eda":
        assignment = agent.bus.history[-2]
        assert assignment.source == "coordinator"
        assert target not in assignment.data["features"]
        assert f"[Coordinator] Completed {specialist}." in capsys.readouterr().out


def test_task_option_requires_coordinated_mode():
    with pytest.raises(SystemExit, match="2"):
        main(["example.csv", "--agent", "pipeline", "--task", "clustering"])


@pytest.mark.parametrize("has_eligible_predictors", [True, False])
def test_coordinated_trace_records_leakage_decision(tmp_path, has_eligible_predictors):
    data = pd.read_csv(DATA / "diabetes_lab03.csv")
    if not has_eligible_predictors:
        data = data[["target"]]
    csv = tmp_path / "conflict.csv"
    data.assign(target_copy=data["target"]).to_csv(csv, index=False)
    trace = tmp_path / "trace.json"
    arguments = [str(csv), "--agent", "coordinated", "--target", "target", "--trace", str(trace)]

    if has_eligible_predictors:
        assert main(arguments) == 0
    else:
        with pytest.raises(SystemExit, match="2"):
            main(arguments)

    evidence = json.loads(trace.read_text(encoding="utf-8"))
    assert {"action": "excluded_predictor", "column": "target_copy",
            "reasons": ["leakage", "target_copy"]} in evidence["decisions"]
    if has_eligible_predictors:
        assert evidence["status"] == "completed"
        assignment = next(event for event in evidence["events"] if event["type"] == "regression_requested")
        assert "target_copy" not in assignment["data"]["features"]
        assert evidence["result"]["algorithm"] == "Lasso"
    else:
        assert evidence["status"] == "needs_review"
        assert [task["agent"] for task in evidence["tasks"]] == ["eda"]
        assert evidence["result"] is None


def test_coordinated_trace_records_failed_task(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("Private diagnostic details")

    monkeypatch.setattr(RegressionAgent, "prepare", fail)
    trace = tmp_path / "failure.json"
    with pytest.raises(SystemExit, match="2"):
        main([str(DATA / "diabetes_lab03.csv"), "--agent", "coordinated", "--target", "target",
              "--trace", str(trace)])

    recorded = trace.read_text(encoding="utf-8")
    evidence = json.loads(recorded)
    assert evidence["status"] == "failed"
    assert evidence["tasks"][-1]["agent"] == "regression"
    assert evidence["tasks"][-1]["status"] == "failed"
    assert evidence["tasks"][-1]["error_code"] == "ValueError"
    assert evidence["tasks"][-1]["duration_seconds"] >= 0
    assert "Private diagnostic details" not in recorded


def test_coordinator_assigns_registered_classifier():
    from ai_framework.agents.blackboard import Blackboard
    from ai_framework.agents.coordinator_agent import CoordinatorAgent
    from ai_framework.agents.registry import AgentDefinition, AgentRegistry
    from ai_framework.agents.run_trace import RunTrace

    registry = AgentRegistry.with_defaults()
    alternate = AgentDefinition("review-classifier", "Review classifier", "classification", frozenset({"classify"}))
    registry.register(alternate)
    assert registry.select("classify").agent_id == "classification"
    with pytest.raises(ValueError, match="cannot perform"):
        registry.select("regress", alternate.agent_id)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(alternate)

    data = pd.read_csv(DATA / "breast_cancer.csv")
    trace = RunTrace()
    coordinator = CoordinatorAgent(Blackboard(data, target="target"), trace.bus, trace=trace, registry=registry)

    agent = coordinator.run(preferred_agent_id=alternate.agent_id)
    evidence = trace.summary(agent=agent)
    assert evidence["status"] == "completed"
    assert evidence["result"]["test_metrics"]["f1_macro"] > 0.9
    assert [task["agent"] for task in evidence["tasks"]] == ["eda", alternate.agent_id]
    assert any(decision.get("agent") == alternate.agent_id for decision in evidence["decisions"])
    requested, completed = evidence["events"][-2:]
    assert (requested["type"], requested["recipient"]) == ("classification_requested", alternate.agent_id)
    assert (completed["type"], completed["source"]) == ("classification_done", alternate.agent_id)


def test_streamlit_coordinator_screens_predictors_using_training_only():
    from ai_framework.data.splitting import dividir_indices
    from ai_framework.ui.controllers.agents_controller import run_coordinated_analysis

    data = pd.read_csv(DATA / "diabetes_lab03.csv")
    _, test = dividir_indices(data, "target", test_size=0.25, random_state=42, stratify=False)
    data["training_target_copy"] = data["target"]
    data.loc[test, "training_target_copy"] = data.loc[test, "target"].iloc[::-1].to_numpy()

    trace = run_coordinated_analysis(data, "diabetes.csv", "target", "auto", language="en")

    assert trace["status"] == "completed"
    assert any(
        decision.get("column") == "training_target_copy" and "target_copy" in decision["reasons"]
        for decision in trace["decisions"]
    )
    assignment = next(event for event in trace["events"] if event["type"] == "regression_requested")
    assert "training_target_copy" not in assignment["data"]["features"]


def test_configured_clustering_comparison_records_operation_and_agent():
    from ai_framework.ui.controllers.agents_controller import run_configured_analysis

    data = pd.read_csv(DATA / "diabetes_lab03.csv").iloc[:40]
    configuration = {
        "features": ["bmi", "bp"], "filas": 40,
        "incluir_categoricas": False, "estandarizar": True,
    }
    execution = run_configured_analysis(
        data, configuration, "clustering", "kmeans_benchmark",
        {"configuration": configuration, "parameters": {"k_min": 2, "k_max": 4}},
        language="en",
    )

    assert execution.output["k"].tolist() == [2, 3, 4]
    request, completed = execution.trace["events"]
    assert request["recipient"] == "clustering"
    assert request["data"]["operation"] == "kmeans_benchmark"
    assert completed["source"] == "clustering"
