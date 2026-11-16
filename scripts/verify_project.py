"""Reproduce the final project flows through the Streamlit entry point."""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from streamlit.testing.v1 import AppTest

from ai_framework.persistence.project import ProjectStore


ROOT = Path(__file__).resolve().parents[1]


def open_app(database):
    app = AppTest.from_file(ROOT / "app.py", default_timeout=180)
    app.session_state["project_database_path"] = database
    return checked(app.run())


def checked(app):
    if app.exception:
        raise RuntimeError(app.exception[0].message)
    return app


def press(app, widgets, label):
    button = next(button for button in widgets if button.label == label)
    return checked(button.click().run())


def navigate(app, route):
    return checked(app.sidebar.button(key=f"nav_{route}").click().run())


def load_csv(app, filename):
    source = app.sidebar.selectbox(key="fuente_archivo")
    source.select_index(source.options.index(filename))
    press(app, app.sidebar.button, "Aplicar fuente")
    if app.session_state["fuente_aplicada"][1] != filename:
        raise RuntimeError(f"Source was not applied: {filename}")


def export_run(store, output, name, trace):
    if trace["status"] != "completed":
        raise RuntimeError(f"{name}: {trace}")
    run = store.get_run(trace["run_id"])
    (output / f"{name}.json").write_text(run["trace"] + "\n", encoding="utf-8")
    if run["csv"]:
        (output / f"{name}.csv").write_text(run["csv"], encoding="utf-8")
    return trace["result"]


def run(output, database):
    app = open_app(database)
    store = ProjectStore(database)
    summary = {}
    load_csv(app, "breast_cancer.csv")
    navigate(app, "datos")
    press(app, app.button, "Aplicar preparación")
    trace = app.session_state["dataset_preparacion"]["trace"]
    export_run(store, output, "preparation", trace)
    navigate(app, "eda")
    if not any(tab.label == "Histogramas" for tab in app.tabs):
        raise RuntimeError("The EDA view did not display its charts.")
    navigate(app, "coordinated_analysis")
    checked(app.button(key="coordinated_run").click().run())
    summary["eda"] = export_run(store, output, "eda", app.session_state["coordinated_result"])

    navigate(app, "configuracion")
    checked(app.radio(key="sup_tarea").set_value("clasificacion").run())
    checked(app.selectbox(key="sup_target").set_value("target").run())
    navigate(app, "lab2_benchmark")
    checked(app.button(key="lab2_benchmark_ejecutar").click().run())
    snapshot = app.session_state["resultado_lab2_experimento"]
    summary["classification"] = export_run(store, output, "classification", snapshot["trace"])
    comparison = snapshot["resultado"]
    winner = comparison.mejores[comparison.mejor_algoritmo]
    summary["classification"].update(
        winner_configuration=winner.metadatos["configuracion"],
        validation_f1_macro=winner.metadatos["puntuacion_validacion"],
        test_f1_macro=winner.metricas["f1"]["macro"],
        confusion_matrix=winner.matriz_confusion.values.tolist(),
    )
    pd.DataFrame([
        {"algorithm": name, "configuration": result.metadatos["configuracion"],
         "validation_f1_macro": result.metadatos["puntuacion_validacion"],
         "test_accuracy": result.metricas["accuracy"], "test_f1_macro": result.metricas["f1"]["macro"]}
        for name, result in comparison.mejores.items()
    ]).to_csv(output / "classification_test.csv", index=False)
    winner.matriz_confusion.to_csv(output / "classification_confusion.csv")

    load_csv(app, "diabetes_lab03.csv")
    navigate(app, "configuracion")
    checked(app.radio(key="sup_tarea").set_value("regresion").run())
    checked(app.selectbox(key="sup_target").set_value("target").run())
    navigate(app, "lab3_benchmark")
    checked(app.button(key="lab3_ejecutar_experimento").click().run())
    snapshot = app.session_state["resultado_lab3_experimento"]
    summary["regression"] = export_run(store, output, "regression", snapshot["trace"])
    comparison = snapshot["resultado"]
    winner = comparison.mejores[comparison.mejor_algoritmo]
    summary["regression"].update(
        winner_configuration=winner.metadatos["configuracion"],
        validation_rmse=winner.metadatos["puntuacion_validacion"],
    )
    pd.DataFrame([
        {"algorithm": name, "configuration": result.metadatos["configuracion"],
         "validation_rmse": result.metadatos["puntuacion_validacion"],
         **{f"test_{key}": result.metricas[key] for key in ("rmse", "mae", "r2")}}
        for name, result in comparison.mejores.items()
    ]).to_csv(output / "regression_test.csv", index=False)

    navigate(app, "agent_community")
    checked(app.selectbox(key="mas_specialty").set_value("clustering").run())
    app.text_input(key="mas_agent_name").set_value("Project clustering reviewer")
    press(app, app.button, "Registrar agente")
    agent = next(item for item in store.registry().list_agents() if item.name == "Project clustering reviewer")
    checked(app.selectbox(key="mas_preferred_clustering").set_value(agent.agent_id).run())
    navigate(app, "coordinated_analysis")
    app.selectbox(key="coordinated_target").set_value("target")
    app.selectbox(key="coordinated_task").set_value("clustering")
    checked(app.button(key="coordinated_run").click().run())
    trace = app.session_state["coordinated_result"]
    summary["clustering"] = export_run(store, output, "clustering", trace)
    request = next(event for event in trace["events"] if event["type"] == "clustering_requested")
    completion = next(event for event in trace["events"] if event["type"] == "clustering_done")
    if request["recipient"] != agent.agent_id or completion["source"] != agent.agent_id:
        raise RuntimeError("The selected specialist did not execute the clustering run.")
    if "target" in request["data"]["features"]:
        raise RuntimeError("The target was included in the clustering inputs.")
    reopened = open_app(database)
    navigate(reopened, "experiment_history")
    if reopened.selectbox(key="history_run").value != trace["run_id"]:
        raise RuntimeError("The new session did not recover the latest run.")
    if not store.get_model(trace["run_id"]):
        raise RuntimeError("The clustering model is missing from the history.")
    summary["checks"] = {
        "registered_agent_id": agent.agent_id, "registered_agent_name": agent.name,
        "excluded_target": "target", "history_reopened": True, "downloadable_model": True,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="nexo-project-") as directory:
        summary = run(args.output, Path(directory) / "workspace.sqlite3")
    summary["environment"] = {
        "python": platform.python_version(),
        "packages": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "xgboost", "streamlit")},
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "data_sha256": {
            name: hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
            for name in ("breast_cancer.csv", "diabetes_lab03.csv")
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Verified dashboard evidence: {args.output}")


if __name__ == "__main__":
    main()
