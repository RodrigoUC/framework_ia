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
