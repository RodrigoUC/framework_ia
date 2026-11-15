"""Verify the coordinated CLI routes requests through the event bus once."""

from pathlib import Path

import pytest

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
