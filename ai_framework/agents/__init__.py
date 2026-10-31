"""AI agents that coordinate the framework with a language model.

Hierarchy, parallel to ``Supervised`` and ``Unsupervised``::

    EDA
    └── Blackboard         (state shared by the agents)
    Agent (ABC)
    ├── EDAAgent           -> perfil, nulos, atípicos, fuga, desbalance
    └── ClusteringAgent    -> compara K-Means y HAC, recomienda k, describe grupos

Uso mínimo::

    pizarra = Blackboard(datos, target="precio")
    bus = EventBus()
    EDAAgent(pizarra, crear_proveedor("gemini")).conectar(bus)
    bus.emitir(Event(DATASET_CARGADO))
"""

from __future__ import annotations

from ai_framework.agents.agent import Agent
from ai_framework.agents.blackboard import Blackboard
from ai_framework.agents.clustering_agent import ClusteringAgent
from ai_framework.agents.eda_agent import EDAAgent
from ai_framework.agents.events import (
    CLUSTERING_LISTO,
    CLUSTERING_SOLICITADO,
    DATASET_CARGADO,
    EDA_LISTO,
    MODELO_LISTO,
    PREGUNTA,
    Event,
    EventBus,
)
from ai_framework.agents.llm import (
    ErrorProveedor,
    ProveedorFalso,
    ProveedorLLM,
    ProveedorOpenAICompat,
    RespuestaLLM,
    crear_proveedor,
)

# Compatibility aliases
Agente = Agent
AgenteClustering = ClusteringAgent
AgenteEDA = EDAAgent
BusEventos = EventBus
Evento = Event
Pizarra = Blackboard

__all__ = [
    "CLUSTERING_LISTO",
    "CLUSTERING_SOLICITADO",
    "DATASET_CARGADO",
    "EDA_LISTO",
    "MODELO_LISTO",
    "PREGUNTA",
    # English names
    "Agent",
    "Blackboard",
    "ClusteringAgent",
    "EDAAgent",
    "ErrorProveedor",
    "Event",
    "EventBus",
    "ProveedorFalso",
    "ProveedorLLM",
    "ProveedorOpenAICompat",
    "RespuestaLLM",
    "crear_proveedor",
    # Bilingual aliases
    "Agente",
    "AgenteClustering",
    "AgenteEDA",
    "BusEventos",
    "Evento",
    "Pizarra",
]
