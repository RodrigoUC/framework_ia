"""Invalidación de sesiones de agentes independiente de los widgets."""

from __future__ import annotations

from ai_framework.ui.models.session_state import huella_datos

CLAVES_EDA = ("agentes_sesion", "error_agentes", "agentes_contexto")
CLAVES_CLUSTERING = (
    "agente_clustering_sesion",
    "error_agente_clustering",
    "agente_clustering_contexto",
)


def _limpiar(estado, claves) -> None:
    for clave in claves:
        estado.pop(clave, None)


def limpiar_resultados_agentes(estado) -> None:
    """Descarta pizarras, hallazgos y conversaciones sin alterar los controles."""
    _limpiar(estado, (*CLAVES_EDA, *CLAVES_CLUSTERING))


def sincronizar_contexto_agentes(estado, datos) -> tuple:
    """Una sesión de agentes sólo es válida para el dataset con que se inició."""
    firma = (estado.get("dataset_identidad"), huella_datos(datos))
    if estado.get("agentes_contexto") != firma:
        _limpiar(estado, CLAVES_EDA)
        estado["agentes_contexto"] = firma
    return firma


def sincronizar_contexto_clustering(estado, datos, configuracion: dict) -> tuple:
    """Cambiar variables, escalado o muestra invalida la partición recomendada."""
    firma = (
        estado.get("dataset_identidad"),
        huella_datos(datos),
        tuple(configuracion["features"]),
        configuracion["incluir_categoricas"],
        configuracion["estandarizar"],
    )
    if estado.get("agente_clustering_contexto") != firma:
        _limpiar(estado, CLAVES_CLUSTERING)
        estado["agente_clustering_contexto"] = firma
    return firma
