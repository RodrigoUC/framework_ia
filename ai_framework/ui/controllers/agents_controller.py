"""Controlador de casos de uso de los agentes de IA de cada sección."""

from __future__ import annotations

import os

import pandas as pd

from ai_framework.agents import (
    CLUSTERING_SOLICITADO,
    DATASET_CARGADO,
    Blackboard,
    ClusteringAgent,
    EDAAgent,
    Event,
    EventBus,
    crear_proveedor,
)
from ai_framework.utils.agents import PROVEEDOR_SIN_LLM, PROVEEDORES_LLM
from ai_framework.utils.results import SugerenciaModelado


def proveedor_por_defecto() -> str:
    """El primer proveedor con key en el entorno; si no hay, sólo reglas."""
    for codigo, proveedor in PROVEEDORES_LLM.items():
        if proveedor.variable_clave and os.environ.get(proveedor.variable_clave):
            return codigo
    return PROVEEDOR_SIN_LLM


def iniciar_agente_eda(datos: pd.DataFrame, etiqueta: str, opciones: dict) -> dict:
    """Ejecuta el agente de exploración y devuelve su sesión, sin la API key."""
    llm = crear_proveedor(opciones["proveedor"], opciones["modelo"], opciones["api_key"])
    # El origen llega al LLM: sólo el nombre del archivo, nunca la ruta local.
    pizarra = Blackboard(datos, target=opciones["target"], origen=etiqueta)
    bus = EventBus()
    agente = EDAAgent(pizarra, llm).conectar(bus)
    bus.emitir(Event(DATASET_CARGADO))
    return {
        "opciones": {k: v for k, v in opciones.items() if k != "api_key"},
        "pizarra": pizarra,
        "agente": agente,
        "bus": bus,
    }


def iniciar_agente_clustering(
    muestra: pd.DataFrame, configuracion: dict, etiqueta: str, opciones: dict
) -> dict:
    """Ejecuta el agente de agrupamiento con la configuración de la sección Clustering."""
    llm = crear_proveedor(opciones["proveedor"], opciones["modelo"], opciones["api_key"])
    pizarra = Blackboard(muestra, origen=etiqueta)
    bus = EventBus()
    agente = ClusteringAgent(
        pizarra,
        llm,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    ).conectar(bus)
    bus.emitir(Event(CLUSTERING_SOLICITADO))
    return {"pizarra": pizarra, "agente": agente, "bus": bus}


def aplicar_sugerencia_modelado(estado, sugerencia: SugerenciaModelado) -> None:
    """Lleva la sugerencia a Configuración; no entrena modelos."""
    estado["sup_tarea"] = sugerencia.tarea
    estado["sup_target"] = sugerencia.target
    estado["sup_features"] = list(sugerencia.features)
    estado["vista_activa"] = "configuracion"
