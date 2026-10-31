"""Run an AI agent on a local CSV file and optionally ask it questions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_framework.agents import (
    CLUSTERING_SOLICITADO,
    DATASET_CARGADO,
    EDA_LISTO,
    ClusteringAgent,
    EDAAgent,
    EventBus,
    Event,
    Blackboard,
    crear_proveedor,
)
from ai_framework.data.sources import CargadorCSV
from ai_framework.utils.agents.agent_config import NOMBRES_PROVEEDORES, PROVEEDOR_SIN_LLM
from ai_framework.utils.data import ConfiguracionCSV


__all__ = [
    "create_parser",
    "crear_parser",
    "run",
    "ejecutar",
    "main",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentes: perfila un CSV, avisa riesgos y propone el modelado."
    )
    parser.add_argument("csv", type=Path, help="CSV local; al LLM sólo llega un resumen.")
    parser.add_argument(
        "--agente", choices=("eda", "clustering"), default="eda",
        help="eda perfila el dataset; clustering compara K-Means y HAC.",
    )
    parser.add_argument("--target", help="Columna objetivo; sin ella sólo se explora.")
    parser.add_argument(
        "--proveedor", choices=list(NOMBRES_PROVEEDORES), default=PROVEEDOR_SIN_LLM,
        help="LLM a usar; 'reglas' trabaja sin red ni API key.",
    )
    parser.add_argument("--modelo", help="Por defecto AGENTES_MODELO o el sugerido.")
    parser.add_argument("--separador", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--interactivo", action="store_true", help="Hacer preguntas al terminar (requiere LLM)."
    )
    return parser


def run(argumentos: argparse.Namespace) -> EDAAgent | ClusteringAgent:
    datos = CargadorCSV.cargar_ruta(
        argumentos.csv,
        ConfiguracionCSV(
            separador=argumentos.separador,
            decimal=argumentos.decimal,
            encoding=argumentos.encoding,
        ),
    )
    pizarra = Blackboard(datos, target=argumentos.target, origen=argumentos.csv.name)
    bus = EventBus()
    llm = crear_proveedor(argumentos.proveedor, argumentos.modelo)
    if argumentos.agente == "clustering":
        agente = ClusteringAgent(pizarra, llm).conectar(bus)
        bus.emitir(Event(CLUSTERING_SOLICITADO))
        return agente
    agente = EDAAgent(pizarra, llm).conectar(bus)
    bus.suscribir(EDA_LISTO, lambda e: print(f">> EDA listo: tarea = {e.datos['tarea']}"))
    bus.emitir(Event(DATASET_CARGADO))
    return agente


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    argumentos = parser.parse_args(argv)
    if argumentos.interactivo and argumentos.proveedor == PROVEEDOR_SIN_LLM:
        parser.error("--interactivo requiere --proveedor distinto de 'reglas'.")
    try:
        agente = run(argumentos)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"No se pudo ejecutar el agente: {exc}\n")
    pizarra = agente.pizarra
    print("\n=== Hallazgos ===")
    for hallazgo in pizarra.hallazgos:
        print(f"[{hallazgo.nivel}] {hallazgo.mensaje}")
    print("\n=== Interpretación ===")
    print(pizarra.mensajes[-1][1])
    if isinstance(agente, EDAAgent) and (sugerencia := agente.sugerir_configuracion()):
        print(f"\nSugerencia: {sugerencia.tarea} sobre {sugerencia.target}; "
              f"excluir {list(sugerencia.excluidas) or 'nada'}.")
    while argumentos.interactivo and (pregunta := input("\nPregunta (Enter para salir): ").strip()):
        try:
            print(agente.preguntar(pregunta))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"No pude responder: {exc}")
    return 0


crear_parser = create_parser
ejecutar = run


if __name__ == "__main__":
    raise SystemExit(main())
