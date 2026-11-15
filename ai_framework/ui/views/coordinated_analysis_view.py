"""Streamlit view for the coordinator's local analysis workflow."""

import json

import streamlit as st

from ai_framework.agents import ProviderError
from ai_framework.ui.controllers.agents_controller import (
    explain_coordinated_run,
    run_coordinated_analysis,
)
from ai_framework.ui.i18n import get_language, t
from ai_framework.ui.models.agents_state import (
    clear_coordinated_chat,
    sync_coordinated_context,
)
from ai_framework.ui.views.agents_view import get_llm_options
from ai_framework.utils.agents import PROVIDER_NAMES, RULES_ONLY

TASK_LABELS = {"regresion": "coordinated_regression", "clasificacion": "coordinated_classification"}
AGENT_LABELS = {
    "regression": "coordinated_regression",
    "classification": "coordinated_classification",
    "clustering": "nav_clustering",
}


def render_coordinated_analysis(data, label: str) -> None:
    st.caption(t("coordinated_intro"))
    target = st.selectbox(t("agent_target", "Variable objetivo"), list(data.columns), index=None,
                          placeholder=t("agent_no_target"), key="coordinated_target", persist_state="session")
    task = st.selectbox(t("coordinated_mode"), ("auto", "clustering"),
                        format_func={"auto": "Auto", "clustering": t("nav_clustering")}.get,
                        key="coordinated_task", persist_state="session")
    sync_coordinated_context(st.session_state, data, target, task)
    use_llm = st.checkbox(t("coordinated_use_llm"), key="coordinated_use_llm", persist_state="session")
    options = get_llm_options()
    configured = options["provider"] != RULES_ONLY
    if use_llm:
        if configured:
            st.caption(" · ".join(filter(None, (PROVIDER_NAMES[options["provider"]], options["model"]))))
        else:
            st.info(t("coordinated_llm_setup"))

    if st.button(t("coordinated_run"), type="primary", key="coordinated_run"):
        clear_coordinated_chat(st.session_state)
        with st.status(t("coordinated_running")) as progress:
            def report(update: str) -> None:
                # Coordinator decisions are rendered from the structured trace below.
                if not update.startswith("[Coordinator]"):
                    progress.write(update)

            trace = run_coordinated_analysis(data, label, target, task,
                                             language=get_language(), on_progress=report)
            st.session_state["coordinated_result"] = trace
            progress.update(state="complete" if trace["status"] == "completed" else "error",
                            label=run_status_label(trace), expanded=False)

    if trace := st.session_state.get("coordinated_result"):
        render_trace(trace)
        if use_llm and configured:
            render_interpretation(trace, options)
        elif not configured:
            clear_coordinated_chat(st.session_state)


def render_interpretation(trace: dict, options: dict) -> None:
    state = st.session_state
    provider = (options["provider"], options["model"])
    if state.get("coordinated_chat_provider") != provider:
        clear_coordinated_chat(state)
        state["coordinated_chat_provider"] = provider

    st.subheader(t("coordinated_interpretation"))
    history = state.setdefault("coordinated_messages", [])
    if not history and state.get("coordinated_chat_error") is None:
        request = {"role": "user", "content": t("coordinated_explain_request")}
        reply = request_explanation(trace, options, [request])
        if reply is not None:
            history.append({"role": "assistant", "content": reply})

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if (error := state.get("coordinated_chat_error")) is not None:
        if isinstance(error, int):
            st.warning(t("coordinated_llm_http", code=error))
        else:
            st.warning(t("coordinated_llm_error"))
            st.caption(t("coordinated_llm_diagnostic", code=error))
        if not history:
            st.button(t("coordinated_llm_retry"), key="coordinated_llm_retry",
                      on_click=state.pop, args=("coordinated_chat_error", None))
    if question := st.chat_input(t("coordinated_question"), key="coordinated_question"):
        request = {"role": "user", "content": question}
        reply = request_explanation(trace, options, [*history, request])
        if reply is not None:
            history.extend((request, {"role": "assistant", "content": reply}))
        st.rerun()


def request_explanation(trace: dict, options: dict, messages: list[dict]) -> str | None:
    try:
        with st.spinner(t("agent_thinking", "Pensando…")):
            answer = explain_coordinated_run(trace, options, messages, language=get_language())
    except Exception as error:  # noqa: BLE001 - provider failures must preserve the local result
        st.session_state["coordinated_chat_error"] = (
            error.status if isinstance(error, ProviderError) else type(error).__name__
        )
        return None
    st.session_state.pop("coordinated_chat_error", None)
    return answer


def run_status_label(trace: dict) -> str:
    return t(f"coordinated_{trace['status']}", code=trace["error_code"])


def render_trace(trace: dict) -> None:
    if trace["status"] == "needs_review":
        st.warning(run_status_label(trace))
    elif trace["status"] == "failed":
        st.error(run_status_label(trace))

    for event in trace["events"]:
        if event["type"] == "eda_done":
            details = event["data"]
            st.write(t("coordinated_detected", task=t(TASK_LABELS[details["task"]]),
                       target=details["target"]))
    for decision in trace["decisions"]:
        if decision["action"] == "assigned":
            st.write(t("coordinated_assigned", agent=t(AGENT_LABELS[decision["agent"]])))
        elif decision["action"] == "excluded_predictor":
            reasons = ", ".join(t(f"coordinated_reason_{reason}") for reason in decision["reasons"])
            st.write(t("coordinated_excluded", column=decision["column"], reasons=reasons))

    if result := trace["result"]:
        st.subheader(t("coordinated_result"))
        if "rows" in result:
            st.write(t("coordinated_profile", rows=result["rows"], columns=result["columns"]))
        else:
            st.metric(t("coordinated_model"), result["algorithm"])
            if "test_metrics" in result:
                st.metric(f"{t('coordinated_validation')} · {result['metric']}", result["validation_score"])
                for metric, value in result["test_metrics"].items():
                    st.metric(f"{t('coordinated_test')} · {metric}", value)
            else:
                st.metric(t("coordinated_clusters"), result["n_clusters"])
                st.metric("Silhouette", result["silhouette"])

    st.download_button(t("coordinated_download"), json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                       file_name="coordinated-analysis.json", mime="application/json")
