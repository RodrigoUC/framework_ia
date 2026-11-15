"""Views of the AI agents of each section; they work on the active dataset."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from ai_framework.ui.i18n import obtener_idioma, t

from ai_framework.agents import Blackboard, EDAAgent
from ai_framework.agents.tools.classification_tools import compare_classifiers
from ai_framework.agents.tools.regression_tools import compare_regressors
from ai_framework.ui.controllers.agents_controller import (
    apply_modeling_suggestion,
    default_provider,
    start_classification_agent,
    start_clustering_agent,
    start_eda_agent,
    start_regression_agent,
)
from ai_framework.ui.controllers.unsupervised_controller import _muestrear
from ai_framework.ui.models.agents_state import (
    sync_clustering_agent_context,
    sync_eda_agent_context,
    sync_supervised_agent_context,
)
from ai_framework.ui.views.navigation import mostrar_faltantes
from ai_framework.ui.views.theme import _mostrar_figura
from ai_framework.utils.agents import LLM_PROVIDERS, PROVIDER_NAMES, RULES_ONLY
from ai_framework.utils.visualization import VisualizadorNoSupervisado

NO_TARGET = "__no_target__"
ICONS = {"critical": ":material/error:", "warning": ":material/warning:", "info": ":material/info:"}
LEVEL_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _provider_names() -> dict[str, str]:
    """Labels resolved once per render, so the dropdown does not depend on the session later."""
    return {
        code: t("agent_rules_only", label) if code == RULES_ONLY else label
        for code, label in PROVIDER_NAMES.items()
    }


def _target_label(no_target: str):
    """Shows the translated placeholder instead of the internal no-target value."""
    return lambda column: no_target if column == NO_TARGET else column


def _llm_controls() -> dict:
    """Provider, model dropdown and API key."""
    left, right = st.columns(2)
    provider = left.selectbox(
        t("agent_llm", "Modelo de lenguaje"),
        list(PROVIDER_NAMES),
        format_func=_provider_names().get,
        key="agent_provider",
    )
    config = LLM_PROVIDERS.get(provider)
    env_model = os.environ.get("AGENTES_MODELO", "").strip()
    model_options = list(config.free_models) if (config and config.free_models) else []
    if config and config.suggested_model and config.suggested_model not in model_options:
        model_options.insert(0, config.suggested_model)
    if env_model and env_model not in model_options:
        model_options.insert(0, env_model)
    if model_options:
        model = right.selectbox(
            t("agent_model", "Modelo"),
            model_options,
            key=f"agent_model_{provider}",
            help=t("agent_model_help", "Vacío: AGENTES_MODELO o la sugerencia del proveedor."),
        )
    else:
        model = right.text_input(
            t("agent_model", "Modelo"),
            value=env_model,
            placeholder=t("agent_model_placeholder", "nombre del modelo"),
            key=f"agent_model_{provider}",
            disabled=(provider == RULES_ONLY),
            help=t("agent_model_help", "Vacío: AGENTES_MODELO o la sugerencia del proveedor."),
        )
    variable = config.key_variable if config else None
    has_key = False
    if variable:
        has_key = bool(os.environ.get(variable))
        if not has_key:
            try:
                has_key = hasattr(st, "secrets") and variable in st.secrets
            except Exception:
                has_key = False
    key_placeholder = (
        t("agent_key_detected", "{variable} detectada", variable=variable)
        if (variable and has_key)
        else (
            t("agent_key_missing", "o defina {variable}", variable=variable)
            if variable
            else t("agent_key_select_provider", "Seleccione un proveedor (ej. Groq o Gemini) para usar una API key")
        )
    )
    api_key = st.text_input(
        t("agent_api_key", "API key"),
        type="password",
        key="agent_api_key",
        placeholder=key_placeholder,
        help=t(
            "agent_api_key_help",
            "Sólo vive en esta sesión. Preferible: .streamlit/secrets.toml.",
        ),
    )
    if config and config.free_models:
        st.caption(
            t(
                "agent_recommended_models",
                "Modelos gratuitos recomendados para {proveedor}: `{modelos}`",
                proveedor=config.label,
                modelos="`, `".join(config.free_models),
            )
        )
    else:
        st.caption(
            t(
                "agent_recommended_models_default",
                "Recomendación: seleccione **Groq** (`openai/gpt-oss-120b`) o **Gemini** (`gemini-2.5-flash`) para habilitar el análisis con LLM.",
            )
        )
    clean_model = model.strip() if model and model.strip() else None
    clean_key = api_key.strip() if api_key and api_key.strip() else None
    return {"provider": provider, "model": clean_model, "api_key": clean_key}


def _start_button() -> bool:
    return st.button(
        t("agent_start", "Iniciar agente"), type="primary", icon=":material/smart_toy:"
    )


def _configure_eda_agent(data: pd.DataFrame) -> dict | None:
    """Returns the configuration only when «Start agent» is pressed."""
    st.session_state.setdefault("agent_provider", default_provider())
    with st.container(border=True):
        st.caption(
            t(
                "agent_eda_description",
                "Perfila el dataset, detecta riesgos y sugiere la configuración de modelado.",
            )
        )
        columns = [NO_TARGET, *map(str, data.columns)]
        current = str(st.session_state.get("sup_target", data.columns[-1]))
        target = st.selectbox(
            t("agent_target", "Variable objetivo"),
            columns,
            index=columns.index(current) if current in columns else len(columns) - 1,
            format_func=_target_label(t("agent_no_target", "(sin variable objetivo)")),
            key="agent_target",
        )
        options = _llm_controls()
        start = _start_button()
    if not start:
        return None
    return {"target": None if target == NO_TARGET else target, **options}


def _configure_clustering_agent() -> dict | None:
    """Uses the variables of the Clustering section; only the button runs clustering."""
    st.session_state.setdefault("agent_provider", default_provider())
    with st.container(border=True):
        st.caption(
            t(
                "agent_clustering_description",
                "Compara K-Means y HAC, recomienda el número de clusters y describe cada grupo.",
            )
        )
        options = _llm_controls()
        start = _start_button()
    return options if start else None


def _render_findings(blackboard: Blackboard) -> None:
    st.subheader(t("agent_findings", "Hallazgos"))
    counts = pd.Series([f.level for f in blackboard.findings], dtype=object).value_counts()
    a, b, c = st.columns(3)
    a.metric(t("agent_critical", "Críticos"), int(counts.get("critical", 0)))
    b.metric(t("agent_warnings", "Avisos"), int(counts.get("warning", 0)))
    c.metric(t("agent_info", "Informativos"), int(counts.get("info", 0)))
    if not blackboard.findings:
        st.success(
            t("agent_no_risks", "El agente no encontró riesgos con sus reglas."),
            icon=":material/check_circle:",
        )
        return
    with st.expander(
        t("agent_findings_detail", "Ver detalle de hallazgos ({total})", total=len(blackboard.findings)),
        expanded=False,
    ):
        for finding in sorted(blackboard.findings, key=lambda f: LEVEL_ORDER[f.level]):
            if finding.level in ("critical", "warning"):
                st.warning(finding.message, icon=ICONS[finding.level])
            else:
                st.info(finding.message, icon=ICONS[finding.level])


def _render_next_step(agent: EDAAgent) -> None:
    suggestion = agent.suggest_configuration()
    st.subheader(t("agent_next_step", "Siguiente paso propuesto"))
    if suggestion is None:
        st.caption(
            t(
                "agent_choose_target",
                "Elija una variable objetivo para que el agente proponga el modelado.",
            )
        )
        return
    task = (
        t("nav_regresion", "Regresión")
        if suggestion.task == "regresion"
        else t("nav_clasificacion", "Clasificación")
    )
    st.markdown(
        t(
            "agent_detected_task",
            "Tarea detectada: **{tarea}** sobre `{target}` con {cantidad} predictoras.",
            tarea=task,
            target=suggestion.target,
            cantidad=len(suggestion.features),
        )
    )
    if suggestion.excluded:
        st.caption(
            t(
                "agent_excludes",
                "Excluye: {columnas}",
                columnas=", ".join(f"`{c}`" for c in suggestion.excluded),
            )
        )
    st.button(
        t("agent_apply", "Aplicar configuración sugerida"),
        key="agent_apply",
        icon=":material/arrow_forward:",
        on_click=apply_modeling_suggestion,
        args=(st.session_state, suggestion),
        help=t(
            "agent_apply_help",
            "Abre Configuración con tarea, target y predictoras ya elegidas; no entrena.",
        ),
    )


def _render_conversation(session: dict, *, key: str, error_key: str) -> None:
    blackboard, agent = session["blackboard"], session["agent"]
    st.subheader(t("agent_conversation", "Conversación"))
    if agent.llm is None:
        st.caption(
            t(
                "agent_configure_llm",
                "Configure un modelo de lenguaje para que el agente interprete y responda preguntas.",
            )
        )
        return
    last_index = len(blackboard.messages) - 1
    for idx, (author, text) in enumerate(blackboard.messages):
        is_user = author == "user"
        with st.chat_message("user" if is_user else "assistant"):
            if is_user:
                st.markdown(text)
            else:
                preview = next((line.strip("#* >-") for line in text.splitlines() if line.strip()), "")
                label = (preview[:70] + "...") if len(preview) > 70 else (preview or t("agent_response", "Respuesta"))
                with st.expander(label, expanded=(idx == last_index)):
                    st.markdown(text)
    question = st.chat_input(t("agent_question", "Pregunte al agente sobre el dataset"), key=key)
    if question:
        # Answers follow the language currently selected, even if it changed after starting.
        agent.language = obtener_idioma()

        def ask(report):
            agent.on_progress = report
            try:
                return agent.ask(question)
            finally:
                agent.on_progress = None

        try:
            _run_with_progress("agent_thinking", "Pensando…", ask)
        except Exception as failure:  # pylint: disable=broad-except
            st.session_state[error_key] = t(
                "agent_llm_error",
                "El modelo de lenguaje no respondió: {detalle}",
                detalle=failure,
            )
        st.rerun()


def _render_recommended_partition(session: dict) -> None:
    blackboard = session["blackboard"]
    suggestion = session["agent"].suggest_clustering()
    result = blackboard.results["cluster_result"]
    st.subheader(t("agent_partition", "Partición recomendada"))
    a, b, c = st.columns(3)
    a.metric(t("agent_algorithm", "Algoritmo"), suggestion.algorithm)
    b.metric(t("agent_clusters", "Clusters"), suggestion.n_clusters)
    c.metric("Silhouette", f"{suggestion.silhouette:.4f}")
    _mostrar_figura(VisualizadorNoSupervisado.clusters(result))
    _mostrar_figura(VisualizadorNoSupervisado.perfiles_cluster(result))
    output = blackboard.datos.loc[result.etiquetas.index].copy()
    output["cluster"] = result.etiquetas
    st.download_button(
        t("agent_download", "Descargar asignaciones"),
        output.to_csv(index=False).encode("utf-8"),
        file_name="clusters_agente.csv",
        mime="text/csv",
        key="agent_clustering_download",
    )
    with st.expander(t("agent_comparison", "Comparación evaluada")):
        kmeans = blackboard.results["kmeans_evaluation"]
        ward = blackboard.results["ward_evaluation"]
        st.dataframe(kmeans, width="stretch")
        _mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(kmeans, "K-Means"))
        if not ward.empty:
            st.dataframe(ward, width="stretch")
            _mostrar_figura(VisualizadorNoSupervisado.curva_evaluacion(ward, "HAC-ward"))


def _run_with_progress(label_key: str, fallback: str, run):
    """Runs an agent inside a live status box; each update appears as its step ends."""
    with st.status(t(label_key, fallback), expanded=True) as status:
        try:
            result = run(status.write)
        except Exception:
            status.update(label=t("agent_failed", "El agente se detuvo."), state="error")
            raise
        status.update(label=t("agent_finished", "El agente terminó."), state="complete", expanded=False)
    return result


def _start_session(session_key: str, label_key: str, fallback: str, start) -> None:
    """Starts an agent in the interface language and keeps its session."""
    session = _run_with_progress(
        label_key, fallback, lambda report: start(language=obtener_idioma(), on_progress=report)
    )
    session["agent"].on_progress = None  # the status box does not survive the rerun
    st.session_state[session_key] = session


def _render_activity(blackboard: Blackboard) -> None:
    if not blackboard.progress:
        return
    with st.expander(t("agent_activity", "Actividad del agente")):
        st.markdown("\n".join(f"- {update}" for _, update in blackboard.progress))


def _show_error(key: str) -> None:
    if error := st.session_state.get(key):
        st.error(error)


def render_eda_agent(data: pd.DataFrame, label: str) -> None:
    """Analyzes only when the button is pressed; opening the view spends no LLM quota."""
    st.caption(
        t(
            "agent_eda_intro",
            "El agente actúa por su cuenta al iniciarse (reglas deterministas) y luego "
            "responde preguntas con el modelo de lenguaje. Al LLM sólo se envía un "
            "resumen del dataset, nunca sus filas.",
        )
    )
    sync_eda_agent_context(st.session_state, data)
    options = _configure_eda_agent(data)
    if options is not None:
        st.session_state.pop("agent_eda_error", None)
        try:
            _start_session(
                "agent_eda_session", "agent_analyzing", "El agente está analizando el dataset…",
                lambda **progress: start_eda_agent(data, label, options, **progress),
            )
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["agent_eda_error"] = t(
                "agent_start_error", "No fue posible iniciar el agente: {detalle}", detalle=error
            )
    _show_error("agent_eda_error")
    session = st.session_state.get("agent_eda_session")
    if session is None:
        st.info(
            t("agent_eda_pending", "Pulse «Iniciar agente» para analizar el dataset."),
            icon=":material/smart_toy:",
        )
        return
    st.divider()
    _render_findings(session["blackboard"])
    _render_activity(session["blackboard"])
    _render_conversation(session, key="agent_eda_question", error_key="agent_eda_error")
    _render_next_step(session["agent"])


def render_clustering_agent(data: pd.DataFrame, config: dict, label: str) -> None:
    """Runs clustering only when the button is pressed, with the Clustering section's variables."""
    if len(data) < 3 or not config["features"]:
        st.warning(t("agent_requires_variables", "Se requieren variables y al menos tres filas."))
        return
    sample = _muestrear(data, config["filas"])
    sync_clustering_agent_context(st.session_state, sample, config)
    options = _configure_clustering_agent()
    if options is not None:
        st.session_state.pop("agent_clustering_error", None)
        try:
            _start_session(
                "agent_clustering_session", "agent_comparing",
                "El agente está comparando agrupamientos…",
                lambda **progress: start_clustering_agent(sample, config, label, options, **progress),
            )
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["agent_clustering_error"] = t(
                "agent_start_error", "No fue posible iniciar el agente: {detalle}", detalle=error
            )
    _show_error("agent_clustering_error")
    session = st.session_state.get("agent_clustering_session")
    if session is None:
        st.info(
            t("agent_clustering_pending", "Pulse «Iniciar agente» para comparar agrupamientos."),
            icon=":material/smart_toy:",
        )
        return
    st.divider()
    _render_findings(session["blackboard"])
    _render_activity(session["blackboard"])
    _render_recommended_partition(session)
    _render_conversation(
        session, key="agent_clustering_question", error_key="agent_clustering_error"
    )


SUPERVISED_FAMILIES = {
    "regression": {
        "start": start_regression_agent,
        "ranking": compare_regressors,
        "description": (
            "agent_regression_description",
            "Compara los regresores con validación cruzada, avisa riesgos y explica el mejor.",
        ),
        "metric_label": "RMSE",
        "test_metrics": (("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")),
    },
    "classification": {
        "start": start_classification_agent,
        "ranking": compare_classifiers,
        "description": (
            "agent_classification_description",
            "Compara los clasificadores con validación cruzada, avisa riesgos y explica el mejor.",
        ),
        "metric_label": "F1 macro",
        "test_metrics": (("f1_macro", "F1 macro"), ("accuracy", "Accuracy"), ("recall_macro", "Recall macro")),
    },
}


def _configure_supervised_agent(family: str) -> dict | None:
    st.session_state.setdefault("agent_provider", default_provider())
    with st.container(border=True):
        st.caption(t(*SUPERVISED_FAMILIES[family]["description"]))
        st.caption(
            t(
                "agent_supervised_note",
                "Usa el target, las predictoras y la partición de Configuración. Compara "
                "los modelos con validación cruzada dentro de train; la prueba sólo "
                "evalúa al ganador.",
            )
        )
        options = _llm_controls()
        start = _start_button()
    return options if start else None


def _render_recommended_model(session: dict, family: str) -> None:
    agent = session["agent"]
    recommendation = agent.suggest_model()
    spec = SUPERVISED_FAMILIES[family]
    st.subheader(t("agent_recommended_model", "Modelo recomendado"))
    columns = st.columns(2 + len(spec["test_metrics"]))
    columns[0].metric(t("agent_algorithm", "Algoritmo"), recommendation.algorithm)
    columns[1].metric(
        t("agent_validation_metric", "{metrica} validación", metrica=spec["metric_label"]),
        f"{recommendation.validation_score:.4f}",
    )
    for column, (key, label) in zip(columns[2:], spec["test_metrics"]):
        column.metric(
            t("agent_test_metric", "{metrica} prueba", metrica=label),
            f"{recommendation.test_metrics[key]:.4f}",
        )
    with st.expander(t("agent_comparison", "Comparación evaluada")):
        ranking = spec["ranking"].function(session["blackboard"])["rows"]
        st.dataframe(pd.DataFrame(ranking), width="stretch")


def _suggest_eda_agent(family: str) -> None:
    """Suggests visiting the EDA agent to review and prepare the dataset configuration."""
    st.info(
        t(
            "agent_suggest_eda_prep",
            "Sugerencia: use el Agente EDA en Exploración de datos para analizar y preparar la configuración del dataset.",
        ),
        icon=":material/explore:",
    )
    if st.button(
        t("agent_go_to_eda", "Ir al Agente EDA"),
        key=f"agent_{family}_goto_eda",
    ):
        st.session_state["vista_activa"] = "agente_eda"
        st.rerun()


def _render_supervised_agent(family, data, config, availability, label) -> None:
    if config is None or not config["features"]:
        st.warning(
            t("agent_configure_supervised", "Seleccione un target y predictoras en Configuración.")
        )
        _suggest_eda_agent(family)
        return
    if availability is not None and not availability.lista:
        mostrar_faltantes(availability)
        _suggest_eda_agent(family)
        return
    session_key, error_key = f"agent_{family}_session", f"agent_{family}_error"
    sync_supervised_agent_context(st.session_state, data, config, family)
    options = _configure_supervised_agent(family)
    if options is not None:
        st.session_state.pop(error_key, None)
        try:
            _start_session(
                session_key, "agent_supervised_running",
                "El agente está comparando modelos con validación cruzada…",
                lambda **progress: SUPERVISED_FAMILIES[family]["start"](
                    data, config, label, options, **progress
                ),
            )
        except Exception as error:  # pylint: disable=broad-except
            msg = str(error)
            if "pliegues" in msg.lower() or "folds" in msg.lower():
                st.session_state[error_key] = (
                    "__insufficient_rows__",
                    t(
                        "agent_insufficient_rows_rec",
                        "El dataset es demasiado pequeño para validación cruzada de 5 pliegues ({detalle}). "
                        "Recomendación: utilice un dataset con más registros por clase, o bien analícelo con el Agente EDA o Clustering.",
                        detalle=msg,
                    ),
                )
            else:
                st.session_state[error_key] = t(
                    "agent_start_error", "No fue posible iniciar el agente: {detalle}", detalle=error
                )
    err_val = st.session_state.get(error_key)
    if isinstance(err_val, tuple) and err_val[0] == "__insufficient_rows__":
        st.warning(err_val[1], icon=":material/lightbulb:")
        _suggest_eda_agent(family)
    else:
        _show_error(error_key)
        if err_val:
            _suggest_eda_agent(family)
    session = st.session_state.get(session_key)
    if session is None:
        st.info(
            t("agent_supervised_pending", "Pulse «Iniciar agente» para comparar modelos."),
            icon=":material/smart_toy:",
        )
        return
    st.divider()
    _render_findings(session["blackboard"])
    _render_activity(session["blackboard"])
    _render_recommended_model(session, family)
    _render_conversation(session, key=f"agent_{family}_question", error_key=error_key)


def render_regression_agent(data: pd.DataFrame, config: dict | None, availability, label: str) -> None:
    """Trains only when the button is pressed, with the regression Settings."""
    _render_supervised_agent("regression", data, config, availability, label)


def render_classification_agent(data: pd.DataFrame, config: dict | None, availability, label: str) -> None:
    """Trains only when the button is pressed, with the classification Settings."""
    _render_supervised_agent("classification", data, config, availability, label)
