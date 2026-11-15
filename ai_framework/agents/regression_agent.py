"""Regression agent built on :class:`SupervisedAgent`."""

from __future__ import annotations

from ai_framework.models.supervised import Regression
from ai_framework.utils.supervised.regression_config import configuraciones_experimento_regresion
from ai_framework.agents.events import REGRESSION_DONE, REGRESSION_REQUESTED
from ai_framework.agents.supervised_agent import SupervisedAgent
from ai_framework.agents.tools.regression_tools import REGRESSION_TOOLS, summarize_regression


class RegressionAgent(SupervisedAgent):
    """Compares the nine regressors and explains the one with the lowest validation RMSE."""

    name = "regression"
    role = (
        "You are the regression agent of a machine learning framework. Use only the "
        "tools to obtain figures; never invent values. Explain, in a few paragraphs, "
        "which model to use and why, how well it predicts (RMSE in target units and R²), "
        "whether it overfits and which variables matter most. The winner was chosen by "
        "cross-validation inside train; the test set only evaluates it. Never suggest "
        "choosing the model by looking at the test set."
    )
    tools = REGRESSION_TOOLS
    task = "regresion"
    requested_event = REGRESSION_REQUESTED
    done_event = REGRESSION_DONE
    result_key = "regression"
    metric = "rmse"

    def __init__(self, blackboard, llm=None, *, simple_feature: str | None = None, **options) -> None:
        super().__init__(blackboard, llm, **options)
        self.simple_feature = simple_feature

    def _candidate_count(self):
        return sum(len(candidates) for candidates in configuraciones_experimento_regresion().values())

    def _run_experiment(self):
        model = Regression(
            dataframe=self.blackboard.datos, target=self.blackboard.target, features=self.features
        )
        experiment = model.experimentar(
            feature_simple=self.simple_feature,
            estrategia="cv",
            folds=self.folds,
            **self._training_options(),
        )
        self.blackboard.results["simple_feature"] = (
            self.simple_feature
            or experiment.metadatos.get("feature_simple")
            or self.features[0]
        )
        return experiment

    def _summarize(self, experiment):
        return summarize_regression(experiment, len(self.features))

    def _recommendation_message(self, summary):
        return self.text(
            "regression_recommendation",
            algorithm=summary["best_algorithm"],
            rmse=summary["validation_rmse"],
            r2=summary["test"]["r2"],
        )

    def _task_rules(self, summary, add):
        thresholds = self.thresholds
        test_r2, train_r2 = summary["test"]["r2"], summary["train"].get("r2")
        if test_r2 <= 0:
            add("critical", self.text("regression_no_better_than_mean", r2=test_r2),
                rule="no_better_than_mean")
        elif test_r2 < thresholds.weak_r2:
            add("warning", self.text("regression_weak", r2=test_r2), rule="weak_model")
        if train_r2 is not None and train_r2 - test_r2 > thresholds.overfitting_gap:
            add("warning", self.text("regression_overfitting", train=train_r2, test=test_r2),
                rule="overfitting")
        rmse, std = summary["validation_rmse"], summary["validation_rmse_std"]
        if rmse and std is not None and std / rmse > thresholds.unstable_cv:
            add("warning", self.text("regression_unstable", std=std, rmse=rmse), rule="unstable_cv")
        test_rmse = summary["test"]["rmse"]
        if rmse and abs(test_rmse - rmse) / rmse > thresholds.validation_test_gap:
            add("warning", self.text("regression_test_gap", test=test_rmse, validation=rmse),
                rule="validation_test_gap")
