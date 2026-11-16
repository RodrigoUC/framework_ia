"""AppTest entry point preserving the app module's package context."""

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from ai_framework.ui.streamlit_app import main

if "project_database_path" not in st.session_state:
    directory = TemporaryDirectory(prefix="nexo-app-test-")
    st.session_state["_test_database_directory"] = directory
    st.session_state["project_database_path"] = Path(directory.name) / "workspace.sqlite3"

main()
