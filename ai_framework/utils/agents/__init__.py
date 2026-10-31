"""Catalogs and thresholds for the AI agents, without Streamlit dependencies."""

from __future__ import annotations

from .agent_config import (
    AGENTE_CLUSTERING,
    AGENTE_EDA,
    NOMBRES_PROVEEDORES,
    PROVEEDOR_SIN_LLM,
    PROVEEDORES_LLM,
    REGLAS_EXCLUSION_EDA,
    ClusteringThresholds,
    ConfiguracionAgente,
    ConfiguracionProveedorLLM,
    EDAThresholds,
    UmbralesClustering,
    UmbralesEDA,
)

__all__ = [
    "AGENTE_CLUSTERING",
    "AGENTE_EDA",
    "NOMBRES_PROVEEDORES",
    "PROVEEDOR_SIN_LLM",
    "PROVEEDORES_LLM",
    "REGLAS_EXCLUSION_EDA",
    "ClusteringThresholds",
    "ConfiguracionAgente",
    "ConfiguracionProveedorLLM",
    "EDAThresholds",
    "UmbralesClustering",
    "UmbralesEDA",
]
