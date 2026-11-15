"""Common base of the regression and classification agents.

Both run the framework's own benchmark (``experimentar``) with cross-validation
inside train, so the test set is used once, to evaluate the winner. They react
to an explicit request or, when chained, to ``EDA_DONE`` for their task.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ai_framework.utils.results import Finding, ModelRecommendation
from ai_framework.utils.agents.agent_config import SupervisedThresholds
from ai_framework.agents.agent import Agent
from ai_framework.agents.events import EDA_DONE, Event

FOLDS = 5
METRIC_LABELS = {"rmse": "RMSE", "f1_macro": "F1 macro"}


class SupervisedAgent(Agent):
    """Runs the benchmark, applies shared rules and recommends a model."""

    task: str  # "regresion" or "clasificacion", as in the rest of the framework
    requested_event: str
    done_event: str
    result_key: str  # where the summary is stored on the blackboard and sent to the LLM
    metric: str

    def __init__(
        self,
        blackboard,
        llm=None,
        *,
        features=None,
        test_size: float = 0.25,
        random_state: int = 42,
        include_categorical: bool = False,
        impute: bool = True,
        standardize: bool = True,
        partition=None,
        folds: int = FOLDS,
        thresholds: SupervisedThresholds | None = None,
        **options,
    ) -> None:
        super().__init__(blackboard, llm, **options)
        self.features = list(features) if features is not None else None
        self.test_size = test_size
        self.random_state = random_state
        self.include_categorical = include_categorical
        self.impute = impute
        self.standardize = standardize
        self.partition = partition
        self.folds = folds
        self.thresholds = thresholds or SupervisedThresholds()

    @property
    def listens_to(self) -> frozenset[str]:  # type: ignore[override]
        return frozenset({self.requested_event, EDA_DONE})

    def accepts(self, event: Event) -> bool:
        """When chained after the EDA agent, act only on datasets of this task."""
        return event.type != EDA_DONE or event.data.get("task") == self.task

    # ---------------------------------------------------------- cycle
    def prepare(self, event: Event) -> None:
        if self.blackboard.target is None:
            raise ValueError(self.text("supervised_no_target"))
        if event.type == EDA_DONE and event.data.get("features"):
            self.features = list(event.data["features"])
        if not self.features:
            self.features = [c for c in self.blackboard.datos.columns if c != self.blackboard.target]
        self.blackboard.results["supervised_features"] = list(self.features)
        self.blackboard.results["supervised_options"] = self._training_options()
        self.report("progress_benchmark_start", candidates=self._candidate_count(), folds=self.folds)
        experiment = self._run_experiment()
        summary = self._summarize(experiment)
        self.blackboard.results[f"{self.result_key}_experiment"] = experiment
        self.blackboard.results[self.result_key] = summary
        self.report(
            "progress_benchmark_done",
            algorithm=summary["best_algorithm"],
            metric=METRIC_LABELS.get(summary["selection_metric"], summary["selection_metric"]),
            score=summary[f"validation_{summary['selection_metric']}"],
        )

    def propose(self, event: Event) -> list[Finding]:
        summary = self.blackboard.results[self.result_key]
        findings: list[Finding] = []

        def add(level, message, **detail):
            findings.append(Finding(self.name, level, message, detail))

        add("info", self._recommendation_message(summary), rule="recommendation")
        if summary["candidates_failed"]:
            add("warning", self.text("supervised_failed", count=summary["candidates_failed"]),
                rule="failed_candidates", count=summary["candidates_failed"])
        rows_per_feature = summary["train_rows"] / max(summary["features"], 1)
        if rows_per_feature < self.thresholds.min_rows_per_feature:
            add("warning", self.text("supervised_few_rows", ratio=rows_per_feature),
                rule="few_rows", rows_per_feature=round(rows_per_feature, 2))
        self._task_rules(summary, add)
        return findings

    def next_events(self, event: Event) -> list[Event]:
        recommendation = self.suggest_model()
        return [
            Event(
                self.done_event,
                {
                    "algorithm": recommendation.algorithm,
                    "metric": recommendation.metric,
                    "validation_score": recommendation.validation_score,
                },
                source=self.name,
            )
        ]

    def suggest_model(self) -> ModelRecommendation | None:
        summary = self.blackboard.results.get(self.result_key)
        if summary is None:
            return None
        return ModelRecommendation(
            self.task,
            summary["best_algorithm"],
            summary["selection_metric"],
            summary[f"validation_{summary['selection_metric']}"],
            dict(summary["test"]),
        )

    # ---------------------------------------------------------- hooks
    def _training_options(self) -> dict[str, Any]:
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "incluir_categoricas": self.include_categorical,
            "imputar": self.impute,
            "estandarizar": self.standardize,
            "particion": self.partition,
        }

    @abstractmethod
    def _candidate_count(self) -> int:
        """How many candidates the benchmark compares, for the progress update."""

    @abstractmethod
    def _run_experiment(self):
        """Runs the framework benchmark for this task."""

    @abstractmethod
    def _summarize(self, experiment) -> dict[str, Any]:
        """JSON-safe summary of the experiment; it is what the LLM sees."""

    @abstractmethod
    def _recommendation_message(self, summary: dict[str, Any]) -> str:
        """Message of the recommendation finding."""

    @abstractmethod
    def _task_rules(self, summary: dict[str, Any], add) -> None:
        """Rules specific to regression or classification."""
