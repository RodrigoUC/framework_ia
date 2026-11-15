"""Run an AI agent on a local CSV file and optionally ask it questions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_framework.agents import (
    CLASSIFICATION_REQUESTED,
    CLUSTERING_REQUESTED,
    DATASET_LOADED,
    EDA_DONE,
    REGRESSION_REQUESTED,
    Agent,
    Blackboard,
    ClassificationAgent,
    ClusteringAgent,
    EDAAgent,
    Event,
    EventBus,
    RegressionAgent,
    SupervisedAgent,
    create_provider,
)
from ai_framework.data.sources import CargadorCSV
from ai_framework.utils.agents.agent_config import PROVIDER_NAMES, RULES_ONLY
from ai_framework.agents.messages import DEFAULT_LANGUAGE, MESSAGES
from ai_framework.utils.data import ConfiguracionCSV

__all__ = [
    "create_parser",
    "crear_parser",
    "run",
    "ejecutar",
    "main",
]

# Value accepted before the translation, kept so existing commands still work.
LEGACY_RULES_ONLY = "reglas"
AGENTS = ("eda", "clustering", "regression", "classification", "pipeline")
SUPERVISED = {
    "regression": (RegressionAgent, REGRESSION_REQUESTED),
    "classification": (ClassificationAgent, CLASSIFICATION_REQUESTED),
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI agents: profile a CSV, flag risks and propose the modeling setup."
    )
    parser.add_argument("csv", type=Path, help="Local CSV; only a summary reaches the LLM.")
    parser.add_argument(
        "--agent", "--agente", dest="agent", choices=AGENTS, default="eda",
        help="eda profiles the dataset; clustering compares K-Means and HAC; regression and "
        "classification compare models with cross-validation; pipeline runs EDA and then "
        "the supervised agent that matches the detected task.",
    )
    parser.add_argument("--target", help="Target column; without it the agent only explores.")
    parser.add_argument(
        "--provider", "--proveedor", dest="provider",
        choices=[*PROVIDER_NAMES, LEGACY_RULES_ONLY], default=RULES_ONLY,
        help=f"LLM to use; '{RULES_ONLY}' works without network access or an API key.",
    )
    parser.add_argument(
        "--model", "--modelo", dest="model", help="Defaults to AGENTES_MODELO or the suggested one."
    )
    parser.add_argument("--separator", "--separador", dest="separator", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--language", "--idioma", dest="language", choices=sorted(MESSAGES), default=DEFAULT_LANGUAGE,
        help="Language of the findings, progress updates and LLM answers.",
    )
    parser.add_argument(
        "--interactive", "--interactivo", dest="interactive", action="store_true",
        help="Ask questions when the analysis ends (requires an LLM).",
    )
    return parser


def run(arguments: argparse.Namespace) -> Agent:
    data = CargadorCSV.cargar_ruta(
        arguments.csv,
        ConfiguracionCSV(
            separador=arguments.separator,
            decimal=arguments.decimal,
            encoding=arguments.encoding,
        ),
    )
    blackboard = Blackboard(data, target=arguments.target, source=arguments.csv.name)
    bus = EventBus()
    llm = create_provider(arguments.provider, arguments.model)
    options = {"language": arguments.language, "on_progress": lambda update: print(f"  - {update}", flush=True)}
    if arguments.agent == "clustering":
        agent = ClusteringAgent(blackboard, llm, **options).connect(bus)
        bus.emit(Event(CLUSTERING_REQUESTED))
        return agent
    if arguments.agent in SUPERVISED:
        agent_class, event_type = SUPERVISED[arguments.agent]
        agent = agent_class(blackboard, llm, **options).connect(bus)
        bus.emit(Event(event_type))
        return agent
    agent = EDAAgent(blackboard, llm, **options).connect(bus)
    bus.subscribe(EDA_DONE, lambda e: print(f">> EDA done: task = {e.data['task']}"))
    if arguments.agent == "pipeline":
        # Both listen to EDA_DONE; only the one matching the detected task acts.
        chained = [agent_class(blackboard, llm, **options).connect(bus) for agent_class, _ in SUPERVISED.values()]
        bus.emit(Event(DATASET_LOADED))
        return next((a for a in chained if a.result_key in blackboard.results), agent)
    bus.emit(Event(DATASET_LOADED))
    return agent


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    arguments = parser.parse_args(argv)
    if arguments.provider == LEGACY_RULES_ONLY:
        arguments.provider = RULES_ONLY
    if arguments.interactive and arguments.provider == RULES_ONLY:
        parser.error(f"--interactive requires a --provider other than '{RULES_ONLY}'.")
    try:
        agent = run(arguments)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"The agent could not run: {exc}\n")
    blackboard = agent.blackboard
    print("\n=== Findings ===")
    for finding in blackboard.findings:
        print(f"[{finding.level}] {finding.message}")
    print("\n=== Interpretation ===")
    for author, text in blackboard.messages:
        print(f"[{author}] {text}")
    if isinstance(agent, EDAAgent) and (suggestion := agent.suggest_configuration()):
        print(f"\nSuggestion: {suggestion.task} on {suggestion.target}; "
              f"exclude {list(suggestion.excluded) or 'nothing'}.")
    if isinstance(agent, SupervisedAgent) and (recommendation := agent.suggest_model()):
        print(f"\nRecommended model: {recommendation.algorithm} "
              f"(validation {recommendation.metric} {recommendation.validation_score}; "
              f"test {recommendation.test_metrics}).")
    while arguments.interactive and (question := input("\nQuestion (Enter to quit): ").strip()):
        try:
            print(agent.ask(question))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Could not answer: {exc}")
    return 0


crear_parser = create_parser
ejecutar = run


if __name__ == "__main__":
    raise SystemExit(main())
