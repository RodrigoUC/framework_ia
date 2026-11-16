"""Access to local project data from an active Streamlit session."""

from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from ai_framework.persistence.project import ProjectStore


def get_project_store() -> ProjectStore:
    if "project_store" not in st.session_state:
        default = Path(__file__).resolve().parents[3] / "output" / "project" / "workspace.sqlite3"
        st.session_state["project_store"] = ProjectStore(Path(st.session_state.get("project_database_path", default)))
    return st.session_state["project_store"]


def active_project_store() -> ProjectStore | None:
    return get_project_store() if get_script_run_ctx(suppress_warning=True) is not None else None
