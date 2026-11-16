"""Exercise agent registration, assignment and durable history in Streamlit."""

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from ai_framework.persistence.project import ProjectStore


ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")
DATA = ENTRYPOINT.parents[1] / "data"


def open_app(database):
    app = AppTest.from_file(ENTRYPOINT, default_timeout=120)
    app.session_state["project_database_path"] = database
    return app.run()


def test_register_assign_and_reopen_history(tmp_path):
    database = tmp_path / "workspace.sqlite3"
    app = open_app(database)
    assert not app.exception
    assert app.sidebar.button(key="nav_project_home").label == "Inicio"
    app.sidebar.selectbox(key="selector_idioma_global").set_value("en").run()
    assert app.sidebar.button(key="nav_project_home").label == "Home"
    app.sidebar.selectbox(key="selector_idioma_global").set_value("es").run()

    app.sidebar.button(key="nav_agent_community").click().run()
    app.selectbox(key="mas_specialty").set_value("clustering").run()
    app.text_input(key="mas_agent_name").set_value("Agrupamiento")
    next(button for button in app.button if button.label == "Registrar agente").click().run()
    assert not app.exception
    agent = next(agent for agent in ProjectStore(database).registry().list_agents() if agent.name == "Agrupamiento")
    selector = app.selectbox(key="mas_preferred_clustering")
    assert selector.options == ["Agrupamiento (Predeterminado)", "Agrupamiento (Registrado)"]
    selector.set_value(agent.agent_id).run()

    data = pd.read_csv(DATA / "diabetes_lab03.csv").iloc[:65]
    app.session_state["fuente_aplicada"] = (data, "diabetes.csv", "local")
    app.run()
    app.sidebar.button(key="nav_coordinated_analysis").click().run()
    app.selectbox(key="coordinated_task").set_value("clustering")
    app.button(key="coordinated_run").click().run()
    assert not app.exception
    assert any("Agrupamiento" in str(item.value) for item in app.markdown)
    trace = app.session_state["coordinated_result"]
    assert trace["status"] == "completed"
    assert trace["tasks"][-1]["agent"] == agent.agent_id
    assert trace["events"][-1]["source"] == agent.agent_id

    reopened = open_app(database)
    reopened.sidebar.button(key="nav_experiment_history").click().run()
    assert not reopened.exception
    assert reopened.selectbox(key="history_run").value == trace["run_id"]
    assert {"history_trace", "history_csv", "history_model"} <= {
        download.key for download in reopened.get("download_button")
    }
    stored = ProjectStore(database).get_run(trace["run_id"])
    assert json.loads(stored["trace"])["result"]["algorithm"] == trace["result"]["algorithm"]
    assert ProjectStore(database).get_model(trace["run_id"])


def test_saved_supervised_artifact_preserves_fitted_preprocessing(tmp_path):
    import pickle

    from ai_framework.models.supervised import Regression
    from ai_framework.ui.controllers.agents_controller import _record_analysis

    data = pd.read_csv(DATA / "diabetes_lab03.csv")
    result = Regression(dataframe=data, target="target", features=["bmi", "bp"]).entrenar(
        algoritmo="RLM", random_state=42,
    )
    store = ProjectStore(tmp_path / "workspace.sqlite3")
    trace = _record_analysis(store, {"status": "completed"}, "diabetes.csv", "regression.train", result)
    model = pickle.loads(store.get_model(trace["run_id"]))

    predicted = model.predict(data.loc[result.y_pred.index, ["bmi", "bp"]])
    assert predicted.tolist() == result.y_pred.tolist()
