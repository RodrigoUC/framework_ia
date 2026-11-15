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
from ai_framework.agents.run_trace import RunTrace
from ai_framework.utils.agents.agent_config import EDA_EXCLUSION_RULES


class ReviewRequiredError(ValueError):
    """The analysis needs input review before training can proceed."""

    run_status = "needs_review"


def predictor_exclusions(blackboard: Blackboard, target: str) -> dict[str, list[str]]:
    """Return the exclusion reasons shared by coordinated and direct workflows."""
    exclusions = {}
    for finding in blackboard.findings:
        rule = finding.detail.get("rule")
        if finding.agent == "eda" and rule in EDA_EXCLUSION_RULES:
            exclusions.setdefault(finding.detail["column"], set()).add(rule)

    data = blackboard.datos
    for column in data:
        if column != target and data[column].equals(data[target]):
            exclusions.setdefault(column, set()).add("target_copy")
    return {column: sorted(reasons) for column, reasons in sorted(exclusions.items())}


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
        trace: RunTrace | None = None,
    ) -> None:
        self.blackboard = blackboard
        self.bus = bus
        self.llm = llm
        self.language = language
        self.on_progress = on_progress
        self.trace = trace

    def run(self, task: str = "auto") -> Agent:
        """Return the agent that completed the request."""
        if task not in {"auto", "clustering"}:
            raise ValueError(f"Unsupported coordinated task: {task!r}.")

        options = {"language": self.language, "on_progress": self.on_progress}
        eda = EDAAgent(self.blackboard, self.llm, **options)
        eda.connect(self.bus)
        target = self.blackboard.target
        self.report("Assigned EDA to profile the dataset.")
        self.dispatch(eda.name, Event(DATASET_LOADED, {"task": task, "target": target}, source=self.name))

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
            features = self.select_predictors(suggestion.features, target)
            if not features:
                self.decision("needs_review", reason="no_eligible_predictors")
                raise ReviewRequiredError("Review required: no eligible predictors remain. Training was not started.")
            specialist = specialist_class(self.blackboard, self.llm, features=features, **options)
            request = specialist.requested_event
            reason = f"EDA identified {specialist.name} for target {target!r}"

        # Connect after EDA finishes so the legacy EDA_DONE subscription cannot start a second run.
        specialist.connect(self.bus)
        self.report(f"Assigned {specialist.name}: {reason}.")
        self.decision("assigned", agent=specialist.name, reason=reason)
        self.dispatch(specialist.name, Event(request, {"target": target, "features": features}, source=self.name))
        self.report(f"Completed {specialist.name}.")
        return specialist

    def select_predictors(self, suggested: tuple[str, ...], target: str) -> list[str]:
        exclusions = predictor_exclusions(self.blackboard, target)
        for column, reasons in exclusions.items():
            self.report(f"Excluded {column!r}: {', '.join(reasons)}.")
            self.decision("excluded_predictor", column=column, reasons=reasons)

        return [column for column in suggested if column != target and column not in exclusions]

    def decision(self, action: str, **details) -> None:
        if self.trace is not None:
            self.trace.decisions.append({"action": action, **details})

    def dispatch(self, agent: str, event: Event) -> None:
        if self.trace is None:
            self.bus.emit(event)
            return
        with self.trace.task(agent):
            self.bus.emit(event)

    def report(self, update: str) -> None:
        self.blackboard.progress.append((self.name, update))
        if self.on_progress is not None:
            self.on_progress(f"[Coordinator] {update}")
