"""Classification agent built on :class:`SupervisedAgent`."""

from __future__ import annotations

from ai_framework.models.supervised import Classification
from ai_framework.utils.supervised.classification_config import ALGORITMOS_CLASIFICACION
from ai_framework.agents.events import CLASSIFICATION_DONE, CLASSIFICATION_REQUESTED
from ai_framework.agents.supervised_agent import SupervisedAgent
from ai_framework.agents.tools.classification_tools import (
    CLASSIFICATION_TOOLS,
    summarize_classification,
)

# One standard configuration per family keeps the agent fast; the full
# hyperparameter benchmark stays in "Evaluación Comparativa".
STANDARD_CONFIGURATIONS = {
    algorithm: [{"nombre": "estandar", "parametros": {}}] for algorithm in ALGORITMOS_CLASIFICACION
}


class ClassificationAgent(SupervisedAgent):
    """Compares the classifiers and explains the one with the best validation F1 macro."""

    name = "classification"
    role = (
        "You are the classification agent of a machine learning framework. Use only the "
        "tools to obtain figures; never invent values. Explain, in a few paragraphs, "
        "which model to use and why, how well it classifies (F1 macro, accuracy and the "
        "classes it misses most), whether it overfits and which variables matter most. "
        "The winner was chosen by cross-validation inside train; the test set only "
        "evaluates it. Never suggest choosing the model by looking at the test set."
    )
    tools = CLASSIFICATION_TOOLS
    task = "clasificacion"
    requested_event = CLASSIFICATION_REQUESTED
    done_event = CLASSIFICATION_DONE
    result_key = "classification"
    metric = "f1_macro"

    def __init__(self, blackboard, llm=None, *, stratify: bool = True, **options) -> None:
        super().__init__(blackboard, llm, **options)
        self.stratify = stratify

    def _training_options(self):
        return {**super()._training_options(), "stratify": self.stratify}

    def _candidate_count(self):
        return len(STANDARD_CONFIGURATIONS)

    def _run_experiment(self):
        model = Classification(
            dataframe=self.blackboard.datos, target=self.blackboard.target, features=self.features
        )
        return model.experimentar(
            algoritmos=list(ALGORITMOS_CLASIFICACION),
            configuraciones=STANDARD_CONFIGURATIONS,
            metrica=self.metric,
            estrategia="cv",
            folds=self.folds,
            **self._training_options(),
        )

    def _summarize(self, experiment):
        return summarize_classification(experiment, len(self.features))

    def _recommendation_message(self, summary):
        return self.text(
            "classification_recommendation",
            algorithm=summary["best_algorithm"],
            validation=summary["validation_f1_macro"],
            test=summary["test"]["f1_macro"],
        )

    def _task_rules(self, summary, add):
        thresholds = self.thresholds
        test = summary["test"]
        if test["accuracy"] <= summary["majority_share"] + thresholds.baseline_margin:
            add("critical", self.text("classification_no_better_than_majority",
                                      accuracy=test["accuracy"], majority=summary["majority_share"]),
                rule="no_better_than_majority")
        train_f1 = summary["train"].get("f1_macro")
        if train_f1 is not None and train_f1 - test["f1_macro"] > thresholds.overfitting_gap:
            add("warning", self.text("classification_overfitting", train=train_f1, test=test["f1_macro"]),
                rule="overfitting")
        if test["accuracy"] - test["f1_macro"] > thresholds.accuracy_f1_gap:
            add("warning", self.text("classification_misleading_accuracy",
                                     accuracy=test["accuracy"], f1=test["f1_macro"]),
                rule="misleading_accuracy")
        for label, scores in summary["classes"].items():
            if scores["recall"] < thresholds.weak_recall:
                add("warning", self.text("classification_weak_class", label=label, recall=scores["recall"]),
                    rule="weak_class", label=label)
        std = summary["validation_f1_macro_std"]
        if std is not None and std > thresholds.unstable_cv_f1:
            add("warning", self.text("classification_unstable", std=std), rule="unstable_cv")
