"""AI agents that coordinate the framework with a language model.

Hierarchy, parallel to ``Supervised`` and ``Unsupervised``::

    EDA
    └── Blackboard         (state shared by the agents)
    Agent (ABC)
    ├── EDAAgent           -> profile, nulls, outliers, leakage, imbalance
    ├── ClusteringAgent    -> compares K-Means and HAC, recommends k, describes groups
    └── SupervisedAgent    (benchmark with CV inside train, shared rules)
        ├── RegressionAgent      -> best regressor by RMSE, overfitting, weak R²
        └── ClassificationAgent  -> best classifier by F1 macro, weak classes, baseline

Minimal usage::

    blackboard = Blackboard(data, target="price")
    bus = EventBus()
    EDAAgent(blackboard, create_provider("gemini")).connect(bus)
    bus.emit(Event(DATASET_LOADED))
"""

from __future__ import annotations

from ai_framework.agents.agent import Agent
from ai_framework.agents.blackboard import Blackboard
from ai_framework.agents.classification_agent import ClassificationAgent
from ai_framework.agents.clustering_agent import ClusteringAgent
from ai_framework.agents.eda_agent import EDAAgent
from ai_framework.agents.events import (
    CLASSIFICATION_DONE,
    CLASSIFICATION_REQUESTED,
    CLUSTERING_DONE,
    CLUSTERING_REQUESTED,
    DATASET_LOADED,
    EDA_DONE,
    MODEL_DONE,
    QUESTION,
    REGRESSION_DONE,
    REGRESSION_REQUESTED,
    Event,
    EventBus,
)
from ai_framework.agents.llm import (
    FakeLLMProvider,
    LLMProvider,
    LLMResponse,
    OpenAICompatibleProvider,
    ProviderError,
    ToolCall,
    create_provider,
)
from ai_framework.agents.regression_agent import RegressionAgent
from ai_framework.agents.supervised_agent import SupervisedAgent

# Compatibility aliases
Agente = Agent
AgenteClustering = ClusteringAgent
AgenteClasificacion = ClassificationAgent
AgenteEDA = EDAAgent
AgenteRegresion = RegressionAgent
BusEventos = EventBus
Evento = Event
Pizarra = Blackboard

__all__ = [
    "CLASSIFICATION_DONE",
    "CLASSIFICATION_REQUESTED",
    "CLUSTERING_DONE",
    "CLUSTERING_REQUESTED",
    "DATASET_LOADED",
    "EDA_DONE",
    "MODEL_DONE",
    "QUESTION",
    "REGRESSION_DONE",
    "REGRESSION_REQUESTED",
    # English names
    "Agent",
    "Blackboard",
    "ClassificationAgent",
    "ClusteringAgent",
    "EDAAgent",
    "Event",
    "EventBus",
    "FakeLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "OpenAICompatibleProvider",
    "ProviderError",
    "RegressionAgent",
    "SupervisedAgent",
    "ToolCall",
    "create_provider",
    # Bilingual aliases
    "Agente",
    "AgenteClasificacion",
    "AgenteClustering",
    "AgenteEDA",
    "AgenteRegresion",
    "BusEventos",
    "Evento",
    "Pizarra",
]
