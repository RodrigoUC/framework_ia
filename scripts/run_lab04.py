"""Compare coordinated and direct workflows on the Lab04 reference cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ai_framework.agents import (
    DATASET_LOADED,
    Blackboard,
    ClassificationAgent,
    CoordinatorAgent,
    EDAAgent,
    Event,
    RegressionAgent,
    SupervisedAgent,
)
from ai_framework.agents.coordinator_agent import (
    ReviewRequiredError,
    predictor_exclusions,
)
from ai_framework.agents.run_trace import RunTrace

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target"
SCENARIOS = {
    "diabetes": ("diabetes_lab03.csv", RegressionAgent, "auto", "plain"),
    "breast_cancer": ("breast_cancer.csv", ClassificationAgent, "auto", "plain"),
    "leakage_with_predictors": ("diabetes_lab03.csv", RegressionAgent, "auto", "copy"),
    "leakage_only": ("diabetes_lab03.csv", RegressionAgent, "auto", "only_copy"),
    "clustering": ("diabetes_lab03.csv", None, "clustering", "plain"),
}


def direct_run(board: Blackboard, specialist_class: type[SupervisedAgent], trace: RunTrace) -> SupervisedAgent:
    """Execute the same analysis stages in a fixed order, without event delivery."""
    eda = EDAAgent(board, language="en")
    with trace.task(eda.name):
        eda.handle(Event(DATASET_LOADED))

    suggestion = eda.suggest_configuration()
    exclusions = predictor_exclusions(board, TARGET)
    trace.decisions.extend(
        {"action": "excluded_predictor", "column": column, "reasons": reasons}
        for column, reasons in exclusions.items()
    )
    features = [column for column in suggestion.features if column not in exclusions]
    if not features:
        trace.decisions.append({"action": "needs_review", "reason": "no_eligible_predictors"})
        raise ReviewRequiredError("No eligible predictors remain.")

    specialist = specialist_class(board, features=features, language="en")
    with trace.task(specialist.name):
        specialist.handle(Event(specialist.requested_event))
    return specialist


def measure(frame, filename, specialist_class, task, mode) -> dict:
    trace = RunTrace()
    try:
        board = Blackboard(frame, target=TARGET, source=filename)
        if mode == "direct":
            agent = direct_run(board, specialist_class, trace)
        else:
            agent = CoordinatorAgent(board, trace.bus, trace=trace).run(task)
    except Exception as error:  # noqa: BLE001 - retain evidence and continue with the other runs
        return trace.summary(error=error)

    evidence = trace.summary(agent=agent)
    if isinstance(agent, SupervisedAgent):
        experiment = board.results[f"{agent.result_key}_experiment"]
        winner = experiment.mejores[experiment.mejor_algoritmo]
        split = [winner.metadatos[key] for key in ("train_indices", "test_indices")]
        evidence["inputs"] = {
            "features": agent.features,
            "split_sha256": sha256(json.dumps(split).encode("utf-8")).hexdigest(),
            "seed": agent.random_state,
            "folds": agent.folds,
        }
        evidence["candidate_errors"] = experiment.metadatos["candidatos_fallidos"]
    return evidence


def run_scenario(name: str, repetition: int) -> list[dict]:
    filename, specialist_class, task, variant = SCENARIOS[name]
    frame = pd.read_csv(ROOT / "data" / filename)
    if variant != "plain":
        frame = frame.assign(target_copy=frame[TARGET])
    if variant == "only_copy":
        frame = frame[[TARGET, "target_copy"]]
    digest = sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()

    modes = ["coordinated", "direct"] if specialist_class else ["coordinated"]
    if repetition % 2 == 0:
        modes.reverse()
    runs = [
        {"scenario": name, "repetition": repetition, "mode": mode, "dataset_sha256": digest,
         **measure(frame, filename, specialist_class, task, mode)}
        for mode in modes
    ]
    expected_status = "needs_review" if variant == "only_copy" else "completed"
    matches = all(run["status"] == expected_status for run in runs)
    if len(runs) == 2:
        matches = matches and all(runs[0].get(field) == runs[1].get(field)
                                  for field in ("inputs", "result", "candidate_errors"))
    if variant != "plain":
        matches = matches and all(
            any(decision["action"] == "excluded_predictor" and decision["column"] == "target_copy"
                for decision in run["decisions"])
            for run in runs
        )
    if variant == "only_copy":
        matches = matches and all([task["agent"] for task in run["tasks"]] == ["eda"] for run in runs)
    if variant == "copy":
        matches = matches and all("target_copy" not in run["inputs"]["features"] for run in runs)
    if task == "clustering":
        matches = matches and any(
            event["type"] == "clustering_requested" and TARGET not in event["data"]["features"]
            for event in runs[0]["events"]
        )
    for run in runs:
        run["matches_expectation"] = matches
    return runs


def summary_row(run: dict) -> dict:
    result = run["result"] or {}
    return {
        "scenario": run["scenario"], "repetition": run["repetition"], "mode": run["mode"],
        "status": run["status"], "seconds": round(run["duration_seconds"], 6),
        "events": len(run["events"]), "tasks_completed": sum(task["status"] == "completed" for task in run["tasks"]),
        "task_errors": sum(task["status"] == "failed" for task in run["tasks"]),
        "candidate_errors": run.get("candidate_errors", 0), "error_code": run["error_code"],
        "algorithm": result.get("algorithm"), "validation_metric": result.get("metric"),
        "validation_score": result.get("validation_score"),
        "test_score": result.get("test_metrics", {}).get(result.get("metric")),
        "silhouette": result.get("silhouette"), "matches_expectation": run["matches_expectation"],
    }


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for JSON and CSV evidence.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append", help="Select cases; defaults to all.")
    options = parser.parse_args(argv)
    if options.repeats < 1:
        parser.error("--repeats must be at least 1.")
    revision = git_value("rev-parse", "HEAD")
    git_status = git_value("status", "--porcelain")
    try:
        options.output.mkdir(parents=True)
    except OSError as error:
        parser.error(str(error))

    runs = []
    for name in dict.fromkeys(options.scenario or SCENARIOS):
        for repetition in range(1, options.repeats + 1):
            print(f"Running {name} ({repetition}/{options.repeats})", flush=True)
            runs.extend(run_scenario(name, repetition))

    evidence = {
        "git_revision": revision, "working_tree_dirty": None if git_status is None else bool(git_status),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "xgboost")},
        "thread_settings": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "provider": "rules", "runs": runs,
    }
    (options.output / "runs.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    rows = [summary_row(run) for run in runs]
    with (options.output / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evidence written to {options.output}")
    return 0 if all(run["matches_expectation"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
