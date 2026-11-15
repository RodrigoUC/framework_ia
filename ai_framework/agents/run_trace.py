"""Structured evidence for a coordinated run."""

from contextlib import contextmanager
from dataclasses import asdict
from time import perf_counter

from ai_framework.agents.clustering_agent import ClusteringAgent
from ai_framework.agents.events import EventBus
from ai_framework.agents.supervised_agent import SupervisedAgent


class RunTrace:
    """Record lifecycle events, decisions and durations without conversation content."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.bus = EventBus()
        self.tasks = []
        self.decisions = []

    @contextmanager
    def task(self, agent: str):
        started_at = perf_counter()
        entry = {"agent": agent, "status": "completed"}
        self.tasks.append(entry)
        try:
            yield
        except Exception as error:
            entry.update(status="failed", error_code=type(error).__name__)
            raise
        finally:
            entry["duration_seconds"] = perf_counter() - started_at

    def summary(self, *, agent=None, error=None) -> dict:
        result = None
        if isinstance(agent, SupervisedAgent):
            result = asdict(agent.suggest_model())
        elif isinstance(agent, ClusteringAgent):
            result = asdict(agent.suggest_clustering())
        elif agent is not None:
            quality = agent.blackboard.results["profile"]["quality"]
            result = {"rows": quality["filas"], "columns": quality["columnas"]}

        fields = {"task", "target", "features", "algorithm", "metric", "validation_score", "n_clusters"}
        return {
            "status": "completed" if error is None else getattr(error, "run_status", "failed"),
            "duration_seconds": perf_counter() - self.started_at,
            "tasks": self.tasks,
            "decisions": self.decisions,
            "events": [
                {"type": event.type, "source": event.source,
                 "data": {key: value for key, value in event.data.items() if key in fields}}
                for event in self.bus.history
            ],
            "result": result,
            "error_code": type(error).__name__ if error is not None else None,
        }
