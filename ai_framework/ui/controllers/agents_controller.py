"""Use-case controller for the AI agents of each section."""

from __future__ import annotations

import os

import pandas as pd

from ai_framework.agents import (
    CLASSIFICATION_REQUESTED,
    CLUSTERING_REQUESTED,
    DATASET_LOADED,
    REGRESSION_REQUESTED,
    Blackboard,
    ClassificationAgent,
    ClusteringAgent,
    EDAAgent,
    Event,
    EventBus,
    RegressionAgent,
    create_provider,
)
from ai_framework.utils.agents import LLM_PROVIDERS, RULES_ONLY
from ai_framework.utils.results import ModelingSuggestion


def default_provider() -> str:
    """The first provider with a key in the environment; rules only if there is none."""
    for code, provider in LLM_PROVIDERS.items():
        if provider.key_variable and os.environ.get(provider.key_variable):
            return code
    return RULES_ONLY


def start_eda_agent(
    data: pd.DataFrame, label: str, options: dict, *, language: str = "es", on_progress=None
) -> dict:
    """Runs the exploration agent and returns its session, without the API key."""
    llm = create_provider(options["provider"], options["model"], options["api_key"])
    # The source reaches the LLM: only the file name, never the local path.
    blackboard = Blackboard(data, target=options["target"], source=label)
    bus = EventBus()
    agent = EDAAgent(blackboard, llm, language=language, on_progress=on_progress).connect(bus)
    bus.emit(Event(DATASET_LOADED))
    return {
        "options": {k: v for k, v in options.items() if k != "api_key"},
        "blackboard": blackboard,
        "agent": agent,
        "bus": bus,
    }


def start_clustering_agent(
    sample: pd.DataFrame, config: dict, label: str, options: dict, *, language: str = "es",
    on_progress=None,
) -> dict:
    """Runs the clustering agent with the settings of the Clustering section."""
    llm = create_provider(options["provider"], options["model"], options["api_key"])
    blackboard = Blackboard(sample, source=label)
    bus = EventBus()
    agent = ClusteringAgent(
        blackboard,
        llm,
        features=config["features"],
        include_categorical=config["incluir_categoricas"],
        standardize=config["estandarizar"],
        language=language,
        on_progress=on_progress,
    ).connect(bus)
    bus.emit(Event(CLUSTERING_REQUESTED))
    return {"blackboard": blackboard, "agent": agent, "bus": bus}


def apply_modeling_suggestion(state, suggestion: ModelingSuggestion) -> None:
    """Carries the suggestion over to Settings; it does not train models."""
    state["sup_tarea"] = suggestion.task
    state["sup_target"] = suggestion.target
    state["sup_features"] = list(suggestion.features)
    state["vista_activa"] = "configuracion"


def _supervised_options(config: dict) -> dict:
    """Maps the shared Settings controls onto the supervised agents' arguments."""
    return {
        "features": config["features"],
        "test_size": config["test_size"],
        "random_state": config["random_state"],
        "include_categorical": config["incluir_categoricas"],
        "impute": config["imputar"],
        "standardize": config["estandarizar"],
        "partition": config.get("particion_global"),
    }


def _start_supervised_agent(
    agent_class, event_type, data, config, label, options, *, language="es", on_progress=None,
    **extra,
) -> dict:
    llm = create_provider(options["provider"], options["model"], options["api_key"])
    blackboard = Blackboard(data, target=config["target"], source=label)
    bus = EventBus()
    agent = agent_class(
        blackboard, llm, language=language, on_progress=on_progress,
        **_supervised_options(config), **extra,
    ).connect(bus)
    bus.emit(Event(event_type))
    return {"blackboard": blackboard, "agent": agent, "bus": bus}


def start_regression_agent(
    data: pd.DataFrame, config: dict, label: str, options: dict, **progress
) -> dict:
    """Runs the regression agent with the target, features and split from Settings."""
    return _start_supervised_agent(
        RegressionAgent, REGRESSION_REQUESTED, data, config, label, options,
        simple_feature=config.get("feature_simple"), **progress,
    )


def start_classification_agent(
    data: pd.DataFrame, config: dict, label: str, options: dict, **progress
) -> dict:
    """Runs the classification agent with the target, features and split from Settings."""
    return _start_supervised_agent(
        ClassificationAgent, CLASSIFICATION_REQUESTED, data, config, label, options,
        stratify=config["estratificar"], **progress,
    )
