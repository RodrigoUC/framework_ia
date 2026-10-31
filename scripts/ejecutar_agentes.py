"""Ejecuta un agente sobre un CSV local y permite hacerle preguntas."""

from __future__ import annotations

import argparse
from pathlib import Path

from framework_ia.agentes import (
    CLUSTERING_SOLICITADO,
    DATASET_CARGADO,
    EDA_LISTO,
    AgenteClustering,
    AgenteEDA,
    BusEventos,
    Evento,
    Pizarra,
    crear_proveedor,
)
from framework_ia.datos.fuentes import CargadorCSV
from framework_ia.utils import NOMBRES_PROVEEDORES, PROVEEDOR_SIN_LLM, ConfiguracionCSV


def crear_parser() -> argparse.ArgumentParser:
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


def ejecutar(argumentos: argparse.Namespace) -> AgenteEDA | AgenteClustering:
    datos = CargadorCSV.cargar_ruta(
        argumentos.csv,
        ConfiguracionCSV(
            separador=argumentos.separador,
            decimal=argumentos.decimal,
            encoding=argumentos.encoding,
        ),
    )
    pizarra = Pizarra(datos, target=argumentos.target, origen=argumentos.csv.name)
    bus = BusEventos()
    llm = crear_proveedor(argumentos.proveedor, argumentos.modelo)
    if argumentos.agente == "clustering":
        agente = AgenteClustering(pizarra, llm).conectar(bus)
        bus.emitir(Evento(CLUSTERING_SOLICITADO))
        return agente
    agente = AgenteEDA(pizarra, llm).conectar(bus)
    bus.suscribir(EDA_LISTO, lambda e: print(f">> EDA listo: tarea = {e.datos['tarea']}"))
    bus.emitir(Evento(DATASET_CARGADO))
    return agente


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    if argumentos.interactivo and argumentos.proveedor == PROVEEDOR_SIN_LLM:
        parser.error("--interactivo requiere --proveedor distinto de 'reglas'.")
    try:
        agente = ejecutar(argumentos)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"No se pudo ejecutar el agente: {exc}\n")
    pizarra = agente.pizarra
    print("\n=== Hallazgos ===")
    for hallazgo in pizarra.hallazgos:
        print(f"[{hallazgo.nivel}] {hallazgo.mensaje}")
    print("\n=== Interpretación ===")
    print(pizarra.mensajes[-1][1])
    if isinstance(agente, AgenteEDA) and (sugerencia := agente.sugerir_configuracion()):
        print(f"\nSugerencia: {sugerencia.tarea} sobre {sugerencia.target}; "
              f"excluir {list(sugerencia.excluidas) or 'nada'}.")
    while argumentos.interactivo and (pregunta := input("\nPregunta (Enter para salir): ").strip()):
        try:
            print(agente.preguntar(pregunta))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"No pude responder: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
