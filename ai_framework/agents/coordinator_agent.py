"""Assign a data analysis request to the existing specialist agents."""

from ai_framework.agents.agent import Agent, ProgressCallback
from ai_framework.agents.blackboard import Blackboard
from ai_framework.agents.classification_agent import ClassificationAgent
from ai_framework.agents.clustering_agent import ClusteringAgent
from ai_framework.agents.eda_agent import EDAAgent
from ai_framework.agents.events import (
    CLUSTERING_REQUESTED,
    DATASET_LOADED,
    Event,
    EventBus,
)
from ai_framework.agents.llm.provider import LLMProvider
from ai_framework.agents.regression_agent import RegressionAgent


class CoordinatorAgent:
    """Profile the data, then assign one specialist through the event bus."""

    name = "coordinator"

    def __init__(
        self,
        blackboard: Blackboard,
        bus: EventBus,
        llm: LLMProvider | None = None,
        *,
        language: str = "en",
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.blackboard = blackboard
        self.bus = bus
        self.llm = llm
        self.language = language
        self.on_progress = on_progress

    def run(self, task: str = "auto") -> Agent:
        """Return the agent that completed the request."""
        if task not in {"auto", "clustering"}:
            raise ValueError(f"Unsupported coordinated task: {task!r}.")

        options = {"language": self.language, "on_progress": self.on_progress}
        eda = EDAAgent(self.blackboard, self.llm, **options)
        eda.connect(self.bus)
        target = self.blackboard.target
        self.report("Assigned EDA to profile the dataset.")
        self.bus.emit(Event(DATASET_LOADED, {"task": task, "target": target}, source=self.name))

        if task == "clustering":
            features = [
                column for column in self.blackboard.datos.select_dtypes(include="number")
                if column != target
            ]
            if not features:
                raise ValueError("Clustering requires a numeric analysis column other than the target.")
            specialist = ClusteringAgent(self.blackboard, self.llm, features=features, **options)
            request = CLUSTERING_REQUESTED
            reason = "clustering was requested explicitly"
        else:
            if target is None:
                self.report("Completed exploration. No target was supplied.")
                return eda
            suggestion = eda.suggest_configuration()
            if suggestion is None:
                raise ValueError(f"EDA did not identify a modeling task for {target!r}.")
            if suggestion.target != target:
                raise ValueError(f"EDA suggested a different target. Run again with --target {suggestion.target!r}.")
            specialist_class = {
                "regresion": RegressionAgent,
                "clasificacion": ClassificationAgent,
            }[suggestion.task]
            features = list(suggestion.features)
            if not features:
                raise ValueError("The coordinated request requires at least one predictor.")
            specialist = specialist_class(self.blackboard, self.llm, features=features, **options)
            request = specialist.requested_event
            reason = f"EDA identified {specialist.name} for target {target!r}"

        # Connect after EDA finishes so the legacy EDA_DONE subscription cannot start a second run.
        specialist.connect(self.bus)
        self.report(f"Assigned {specialist.name}: {reason}.")
        self.bus.emit(Event(request, {"target": target, "features": features}, source=self.name))
        self.report(f"Completed {specialist.name}.")
        return specialist

    def report(self, update: str) -> None:
        self.blackboard.progress.append((self.name, update))
        if self.on_progress is not None:
            self.on_progress(f"[Coordinator] {update}")
