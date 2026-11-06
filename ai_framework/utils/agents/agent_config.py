"""Catalog of agents and LLM providers shared by the core, scripts and UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProviderConfig:
    """Endpoint compatible with the OpenAI chat API, with no Streamlit dependency."""

    label: str
    base_url: str
    key_variable: str | None
    suggested_model: str  # free catalogs change; check with the provider


@dataclass(frozen=True)
class AgentConfig:
    label: str
    description: str


@dataclass(frozen=True)
class EDAThresholds:
    """Limits of the exploration agent's proactive rules."""

    nulls: float = 30.0  # % per column
    leakage: float = 0.98  # |r| between a predictor and the target
    imbalance: float = 0.10  # share of the minority class
    min_identifier_rows: int = 20
    max_classification_classes: int = 10  # more numeric values -> regression


@dataclass(frozen=True)
class ClusteringThresholds:
    """Limits of the clustering agent's proactive rules."""

    weak_silhouette: float = 0.25  # below this there is no substantial structure
    small_cluster: float = 0.05  # minimum share of rows per cluster
    nulls: float = 30.0  # % per column, imputed before clustering
    k_max: int = 10
    min_identifier_rows: int = 20


@dataclass(frozen=True)
class SupervisedThresholds:
    """Limits of the regression and classification agents' proactive rules."""

    overfitting_gap: float = 0.15  # train score minus test score (R² or F1 macro)
    weak_r2: float = 0.30  # test R² below this explains little variance
    unstable_cv: float = 0.25  # relative std of the CV RMSE (regression)
    unstable_cv_f1: float = 0.10  # std of the CV F1 macro (classification)
    validation_test_gap: float = 0.25  # relative difference between CV and test RMSE
    min_rows_per_feature: int = 10
    weak_recall: float = 0.50  # a class detected less often than this
    accuracy_f1_gap: float = 0.15  # accuracy inflated by imbalance
    baseline_margin: float = 0.02  # accuracy must beat the majority class by this


RULES_ONLY = "rules"
LLM_PROVIDERS = {
    "gemini": LLMProviderConfig(
        "Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY", "gemini-2.5-flash",
    ),
    "groq": LLMProviderConfig(
        "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile",
    ),
    "openrouter": LLMProviderConfig(
        "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "",
    ),
    "ollama": LLMProviderConfig(
        "Ollama (local)", "http://localhost:11434/v1", None, "llama3.2",
    ),
}
PROVIDER_NAMES = {
    RULES_ONLY: "No LLM (rules only)",
    **{code: provider.label for code, provider in LLM_PROVIDERS.items()},
}
EDA_AGENT = AgentConfig(
    "EDA agent",
    "Profiles the dataset, detects risks and suggests the modeling setup.",
)
CLUSTERING_AGENT = AgentConfig(
    "Clustering agent",
    "Compares K-Means and HAC, recommends the number of clusters and describes each group.",
)
REGRESSION_AGENT = AgentConfig(
    "Regression agent",
    "Compares the regressors with cross-validation, flags risks and explains the best one.",
)
CLASSIFICATION_AGENT = AgentConfig(
    "Classification agent",
    "Compares the classifiers with cross-validation, flags risks and explains the best one.",
)
EDA_EXCLUSION_RULES = frozenset({"constant", "identifier", "leakage"})

# Compatibility aliases
UmbralesClustering = ClusteringThresholds
UmbralesEDA = EDAThresholds
UmbralesSupervisado = SupervisedThresholds
