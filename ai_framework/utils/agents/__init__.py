"""Catalogs and thresholds for the AI agents, without Streamlit dependencies."""

from __future__ import annotations

from .agent_config import (
    CLASSIFICATION_AGENT,
    CLUSTERING_AGENT,
    EDA_AGENT,
    EDA_EXCLUSION_RULES,
    LLM_PROVIDERS,
    PROVIDER_NAMES,
    REGRESSION_AGENT,
    RULES_ONLY,
    AgentConfig,
    ClusteringThresholds,
    EDAThresholds,
    LLMProviderConfig,
    SupervisedThresholds,
    UmbralesClustering,
    UmbralesEDA,
    UmbralesSupervisado,
)

__all__ = [
    "CLASSIFICATION_AGENT",
    "CLUSTERING_AGENT",
    "EDA_AGENT",
    "EDA_EXCLUSION_RULES",
    "LLM_PROVIDERS",
    "PROVIDER_NAMES",
    "REGRESSION_AGENT",
    "RULES_ONLY",
    "AgentConfig",
    "ClusteringThresholds",
    "EDAThresholds",
    "LLMProviderConfig",
    "SupervisedThresholds",
    "UmbralesClustering",
    "UmbralesEDA",
    "UmbralesSupervisado",
]
