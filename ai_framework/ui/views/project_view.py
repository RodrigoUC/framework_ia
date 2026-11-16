"""Home, agent community, and saved experiment views."""

import json

import streamlit as st

from ai_framework.agents.registry import SPECIALISTS
from ai_framework.ui.i18n import t
from ai_framework.ui.models.project_state import active_project_store, get_project_store


def render_project_home(app_name: str) -> None:
    store = active_project_store()
    agents = store.registry().list_agents() if store else ()
    st.markdown(f"# {app_name}")
    st.caption(t("project_intro"))
    st.metric(t("project_active_agents"), sum(agent.active for agent in agents) + 1)
    st.metric(t("project_saved_runs"), len(store.list_runs()) if store else 0)
    st.info(t("project_start"))


def display_agent(agent) -> str:
    if agent.agent_id == agent.specialty:
        return f"{t(f'project_specialty_{agent.specialty}')} ({t('project_builtin')})"
    return f"{agent.name} ({t('project_registered')})"


def render_agent_community() -> None:
    store = get_project_store()
    st.header(t("nav_agent_community"))
    st.subheader(t("project_create_agent"))
    specialties = {family: t(f"project_specialty_{family}") for family in SPECIALISTS}
    specialty = st.selectbox(
        t("project_specialty"), tuple(SPECIALISTS),
        format_func=specialties.__getitem__, key="mas_specialty",
    )
    capabilities = sorted(SPECIALISTS[specialty][1])
    capability_labels = {value: t(f"project_capability_{value}") for value in capabilities}
    with st.form("mas_registration"):
        name = st.text_input(t("project_agent_name"), key="mas_agent_name")
        selected = st.multiselect(
            t("project_capabilities"), capabilities, default=capabilities,
            format_func=capability_labels.__getitem__, key=f"mas_capabilities_{specialty}",
        )
        register = st.form_submit_button(t("project_register"))
    if register:
        if not name.strip() or not selected:
            st.warning(t("project_registration_required"))
        else:
            try:
                store.register(name, specialty, frozenset(selected))
            except ValueError:
                st.error(t("project_duplicate_agent"))
            else:
                st.success(t("project_agent_registered", name=name.strip()))

    agents = store.registry().list_agents()
    st.dataframe([
        {
            t("project_agent_name"): t("project_coordinator"),
            t("project_specialty"): t("project_coordinator"),
            t("project_capabilities"): t("project_capability_coordinate"),
            t("project_state"): t("project_active"),
        },
        *[
            {
                t("project_agent_name"): display_agent(agent),
                t("project_specialty"): t(f"project_specialty_{agent.specialty}"),
                t("project_capabilities"): ", ".join(t(f"project_capability_{value}") for value in sorted(agent.capabilities)),
                t("project_state"): t("project_active") if agent.active else t("project_inactive"),
            }
            for agent in agents
        ],
    ], hide_index=True, width="stretch")

    st.subheader(t("project_assignment"))
    st.caption(t("project_assignment_help"))
    preferences = {}
    for family in SPECIALISTS:
        eligible = {
            agent.agent_id: display_agent(agent)
            for agent in agents if agent.active and agent.specialty == family
        }
        preferences[family] = st.selectbox(
            t(f"project_specialty_{family}"), tuple(eligible),
            format_func=eligible.__getitem__,
            key=f"mas_preferred_{family}", persist_state="session",
        )
    st.session_state["mas_preferences"] = preferences
    st.subheader(t("project_tasks"))
    show_run_evidence(store, key="community")


def render_experiment_history() -> None:
    st.header(t("nav_experiment_history"))
    show_run_evidence(get_project_store(), key="history")


def show_run_evidence(store, *, key: str) -> None:
    runs = store.list_runs()
    if not runs:
        st.info(t("project_no_runs"))
        return
    choices = {run["id"]: run for run in runs}

    def label(run_id: str) -> str:
        run = choices[run_id]
        return f"{run['created_at'][:19].replace('T', ' ')} UTC · {run['source']} · {run['operation']}"

    selected = st.selectbox(t("project_run"), tuple(choices), format_func=label, key=f"{key}_run")
    run = store.get_run(selected)
    trace = json.loads(run["trace"])
    st.write(t(f"coordinated_{run['status']}", code=trace.get("error_code")))
    if trace["tasks"]:
        names = {agent.agent_id: display_agent(agent) for agent in store.registry().list_agents()}
        st.dataframe([
            {
                t("project_agent_name"): names.get(task["agent"], task["agent"]),
                t("project_state"): t(f"project_run_{task['status']}"),
                t("project_duration"): round(task["duration_seconds"], 3),
            }
            for task in trace["tasks"]
        ], hide_index=True, width="stretch")
    if trace["events"]:
        with st.expander(t("project_messages")):
            st.json(trace["events"], expanded=False)
    if trace["result"]:
        with st.expander(t("coordinated_result")):
            st.json(trace["result"], expanded=False)
    st.download_button(
        t("coordinated_download"), run["trace"], file_name=f"{selected}.json",
        mime="application/json", key=f"{key}_trace",
    )
    if run["csv"] is not None:
        st.download_button(
            t("project_download_csv"), run["csv"], file_name=f"{selected}.csv",
            mime="text/csv", key=f"{key}_csv",
        )
    if model := store.get_model(selected):
        st.download_button(
            t("project_download_model"), model, file_name=f"{selected}.pkl",
            mime="application/octet-stream", key=f"{key}_model",
        )
