"""Agentes que coordinan el framework con un modelo de lenguaje.

Jerarquía, en paralelo a ``Supervisado`` y ``NoSupervisado``::

    EDA
    └── Pizarra      (estado compartido por los agentes)
    Agente (ABC)
    ├── AgenteEDA         -> perfil, nulos, atípicos, fuga, desbalance
    └── AgenteClustering  -> compara K-Means y HAC, recomienda k, describe grupos

Uso mínimo::

    pizarra = Pizarra(datos, target="precio")
    bus = BusEventos()
    AgenteEDA(pizarra, crear_proveedor("gemini")).conectar(bus)
    bus.emitir(Evento(DATASET_CARGADO))
"""

from .agrupamiento import AgenteClustering
from .base import Agente
from .eventos import (
    CLUSTERING_LISTO,
    CLUSTERING_SOLICITADO,
    DATASET_CARGADO,
    EDA_LISTO,
    MODELO_LISTO,
    PREGUNTA,
    BusEventos,
    Evento,
)
from .exploracion import AgenteEDA
from .llm import (
    ErrorProveedor,
    ProveedorFalso,
    ProveedorLLM,
    ProveedorOpenAICompat,
    RespuestaLLM,
    crear_proveedor,
)
from .pizarra import Pizarra

__all__ = [
    "CLUSTERING_LISTO",
    "CLUSTERING_SOLICITADO",
    "DATASET_CARGADO",
    "EDA_LISTO",
    "MODELO_LISTO",
    "PREGUNTA",
    "Agente",
    "AgenteClustering",
    "AgenteEDA",
    "BusEventos",
    "ErrorProveedor",
    "Evento",
    "Pizarra",
    "ProveedorFalso",
    "ProveedorLLM",
    "ProveedorOpenAICompat",
    "RespuestaLLM",
    "crear_proveedor",
]
