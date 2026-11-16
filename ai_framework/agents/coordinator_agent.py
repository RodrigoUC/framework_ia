"""Assign a data analysis request to the existing specialist agents."""

from ai_framework.agents.agent import Agent, ProgressCallback
from ai_framework.agents.blackboard import Blackboard
from ai_framework.agents.analysis_request import AnalysisRequest, AnalysisResult
from ai_framework.agents.events import (
    CLUSTERING_REQUESTED,
    DATASET_LOADED,
    Event,
    EventBus,
)
from ai_framework.agents.llm.provider import LLMProvider
from ai_framework.agents.registry import AgentFactory, AgentRegistry
from ai_framework.agents.run_trace import RunTrace
from ai_framework.utils.agents.agent_config import EDA_EXCLUSION_RULES


class ReviewRequiredError(ValueError):
    """The analysis needs input review before training can proceed."""

    run_status = "needs_review"


def predictor_exclusions(blackboard: Blackboard, target: str, *, eda_id: str = "eda") -> dict[str, list[str]]:
    """Return the exclusion reasons shared by coordinated and direct workflows."""
    exclusions = {}
    for finding in blackboard.findings:
        rule = finding.detail.get("rule")
        if finding.agent == eda_id and rule in EDA_EXCLUSION_RULES:
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
        registry: AgentRegistry | None = None,
    ) -> None:
        self.blackboard = blackboard
        self.bus = bus
        self.llm = llm
        self.language = language
        self.on_progress = on_progress
        self.trace = trace
        self.registry = registry if registry is not None else AgentRegistry.with_defaults()
        self.factory = AgentFactory()

    def run(
        self, task: str = "auto", *, preferred_agent_id: str | None = None,
        training_only_screening: bool = False, agent_preferences: dict[str, str] | None = None,
    ) -> Agent:
        """Return the agent that completed the request."""
        if task not in {"auto", "clustering"}:
            raise ValueError(f"Unsupported coordinated task: {task!r}.")

        options = {"language": self.language, "on_progress": self.on_progress}
        preferences = agent_preferences or {}
        eda_definition = self.registry.select("profile", preferences.get("eda"))
        eda = self.factory.create(eda_definition, self.blackboard, self.llm, **options)
        target = self.blackboard.target
        self.report("Assigned EDA to profile the dataset.")
        self.dispatch(eda, Event(DATASET_LOADED, {"task": task, "target": target}, source=self.name, recipient=eda.name))

        if task == "clustering":
            preferred_agent_id = preferred_agent_id or preferences.get("clustering")
            features = [
                column for column in self.blackboard.datos.select_dtypes(include="number")
                if column != target
            ]
            if not features:
                raise ValueError("Clustering requires a numeric analysis column other than the target.")
            capability = "cluster"
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
            capability = {"regresion": "regress", "clasificacion": "classify"}[suggestion.task]
            specialty = {"regresion": "regression", "clasificacion": "classification"}[suggestion.task]
            preferred_agent_id = preferred_agent_id or preferences.get(specialty)
            if training_only_screening:
                candidate = self.registry.select(capability, preferred_agent_id)
                features = self.screen_training_predictors(candidate, eda_definition, target)
            else:
                features = self.select_predictors(suggestion.features, target, eda_id=eda.name)
            if not features:
                self.decision("needs_review", reason="no_eligible_predictors")
                raise ReviewRequiredError("Review required: no eligible predictors remain. Training was not started.")
            reason = f"EDA identified {capability} capability for target {target!r}"

        definition = self.registry.select(capability, preferred_agent_id)
        specialist = self.factory.create(definition, self.blackboard, self.llm, features=features, **options)
        request = CLUSTERING_REQUESTED if task == "clustering" else specialist.requested_event

        self.report(f"Assigned {specialist.name}: {reason}.")
        self.decision("assigned", agent=specialist.name, name=specialist.display_name, specialty=definition.specialty, reason=reason)
        self.dispatch(specialist, Event(request, {"target": target, "features": features}, source=self.name, recipient=specialist.name))
        self.report(f"Completed {specialist.name}.")
        return specialist

    def run_configured(self, request: AnalysisRequest, *, preferred_agent_id: str | None = None) -> AnalysisResult:
        """Assign an explicit operation and return its output with the same run's trace."""
        capability = {
            "classification": "classify", "regression": "regress", "clustering": "cluster", "eda": "profile",
        }[request.specialty]
        definition = self.registry.select(capability, preferred_agent_id)
        agent = self.factory.create(definition, self.blackboard, self.llm, language=self.language)
        self.decision(
            "assigned", agent=agent.name, name=agent.display_name, specialty=definition.specialty,
            reason=f"The caller requested {request.specialty} {request.operation}.",
        )
        result = None
        summary = {}

        def execute(event: Event) -> None:
            nonlocal result, summary
            if event.recipient != agent.name:
                return
            result = agent.execute_configured(request)
            algorithm = getattr(result, "mejor_algoritmo", getattr(result, "algoritmo", None))
            summary = {"algorithm": algorithm} if algorithm else {"operation": request.operation}
            self.bus.emit(Event(f"{request.specialty}_done", summary, source=agent.name))

        options = request.options
        strategy = options.get("estrategia")
        if request.specialty == "classification" and request.operation == "benchmark" and strategy is None:
            strategy = "holdout"
        self.dispatch(agent, Event(
            f"{request.specialty}_{request.operation}_requested",
            {"task": request.specialty, "target": self.blackboard.target, "features": request.features,
             "operation": request.operation, "strategy": strategy},
            source=self.name, recipient=agent.name,
        ), handler=execute)
        evidence = self.trace.summary() if self.trace is not None else {}
        evidence["result"] = summary
        return AnalysisResult(result, evidence)

    def screen_training_predictors(self, specialist_definition, eda_definition, target: str) -> list[str]:
        """Apply predictor rules using the same training split as the automatic specialist."""
        candidate = self.factory.create(specialist_definition, self.blackboard)
        model = candidate.model_class(dataframe=self.blackboard.datos, target=target)
        base = model._datos_supervisados()
        train, _, _ = model._indices(
            base, particion=None, test_size=candidate.test_size, random_state=candidate.random_state,
            stratify=getattr(candidate, "stratify", False),
        )
        training = Blackboard(self.blackboard.datos.loc[train], target=target)
        profiler = self.factory.create(eda_definition, training, language=self.language)
        profiler.handle(Event(DATASET_LOADED))
        exclusions = predictor_exclusions(training, target, eda_id=profiler.name)
        return self.record_exclusions(
            tuple(base.columns), target, exclusions,
        )

    def select_predictors(self, suggested: tuple[str, ...], target: str, *, eda_id: str = "eda") -> list[str]:
        exclusions = predictor_exclusions(self.blackboard, target, eda_id=eda_id)
        return self.record_exclusions(suggested, target, exclusions)

    def record_exclusions(self, candidates, target, exclusions) -> list[str]:
        for column, reasons in exclusions.items():
            self.report(f"Excluded {column!r}: {', '.join(reasons)}.")
            self.decision("excluded_predictor", column=column, reasons=reasons)

        return [column for column in candidates if column != target and column not in exclusions]

    def decision(self, action: str, **details) -> None:
        if self.trace is not None:
            self.trace.decisions.append({"action": action, **details})

    def dispatch(self, agent: Agent, event: Event, *, handler=None) -> None:
        agent.bus = self.bus
        handler = handler if handler is not None else agent.handle
        self.bus.subscribe(event.type, handler)
        try:
            if self.trace is None:
                self.bus.emit(event)
            else:
                with self.trace.task(agent.name):
                    self.bus.emit(event)
        finally:
            self.bus.unsubscribe(event.type, handler)

    def report(self, update: str) -> None:
        self.blackboard.progress.append((self.name, update))
        if self.on_progress is not None:
            self.on_progress(f"[Coordinator] {update}")
