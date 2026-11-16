"""Exploratory analysis agent built on :class:`Agent`."""

from __future__ import annotations

from ai_framework.utils.results import Finding, ModelingSuggestion
from ai_framework.utils.agents.agent_config import EDA_EXCLUSION_RULES, EDAThresholds
from ai_framework.agents.agent import Agent
from ai_framework.agents.events import DATASET_LOADED, EDA_DONE, Event
from ai_framework.agents.tools.eda_tools import (
    EDA_TOOLS,
    dataset_profile,
    looks_like_identifier,
)


class EDAAgent(Agent):
    """Profiles the dataset when it is loaded, flags problems and enables modeling."""

    name = "eda"
    role = (
        "You are the exploratory data analysis (EDA) specialist of a machine learning framework, "
        "assisting the user directly in the Exploration section. "
        "Always use your tools (`dataset_profile`, `correlations`, `outliers`, `column_frequencies`, "
        "`propose_modeling_setup`) to inspect the data and obtain exact figures before drawing "
        "conclusions—never invent values. "
        "Proactively help the user interpret the components of this section: dataset dimensions, "
        "missing values, data types, collinearity/correlations, outliers, target distribution or "
        "class imbalance, and data leakage risks. "
        "If the user explicitly requests a specific modeling task (such as regression instead of "
        "classification) or a different target/feature subset, you MUST call "
        "`propose_modeling_setup` BEFORE answering so the UI 'Apply suggested configuration' "
        "button updates accordingly. "
        "Never suggest using the test set to make modeling or preprocessing decisions."
    )
    tools = EDA_TOOLS
    listens_to = frozenset({DATASET_LOADED})

    def __init__(
        self, blackboard, llm=None, *, thresholds: EDAThresholds | None = None, **options
    ) -> None:
        super().__init__(blackboard, llm, **options)
        self.thresholds = thresholds or EDAThresholds()

    def execute_configured(self, request):
        if request.operation == "prepare":
            prepared = self.blackboard.preparar_dataset(**request.options)
            return prepared, self.blackboard.preparacion
        if request.operation != "profile":
            raise ValueError(f"Unsupported EDA operation: {request.operation!r}.")
        self.prepare(Event(DATASET_LOADED))
        return self.blackboard

    def prepare(self, event: Event) -> None:
        quality = dataset_profile.function(self.blackboard)["quality"]
        self.report(
            "progress_profile",
            rows=quality["filas"],
            columns=quality["columnas"],
            nulls=quality["nulos"],
            duplicates=quality["duplicados"],
        )

    def propose(self, event: Event) -> list[Finding]:
        profile = self.blackboard.results["profile"]
        quality = profile["quality"]
        findings: list[Finding] = []

        def add(level, message, **detail):
            findings.append(Finding(self.name, level, message, detail))

        if quality["duplicados"]:
            add("warning", self.text("eda_duplicates", count=quality["duplicados"]),
                rule="duplicates", duplicates=quality["duplicados"])
        for column in profile["columns"]:
            name = column["columna"]
            if name == self.blackboard.target:
                continue
            if column["porcentaje_nulos"] > self.thresholds.nulls:
                add("warning", self.text("eda_nulls", column=name, percent=column["porcentaje_nulos"]),
                    rule="nulls", column=name)
            if column["valores_unicos"] <= 1:
                add("warning", self.text("eda_constant", column=name),
                    rule="constant", column=name)
            elif looks_like_identifier(
                self.blackboard.datos[name], self.thresholds.min_identifier_rows
            ):
                add("warning", self.text("eda_identifier", column=name),
                    rule="identifier", column=name)

        target = profile.get("target")
        if target is None:
            add("info", self.text("eda_no_target"))
            return findings
        for column, r in target.get("correlations", {}).items():
            if abs(r) >= self.thresholds.leakage:
                add("critical", self.text("eda_leakage", column=column, r=r),
                    rule="leakage", column=column, r=r)
        classes = target.get("classes", {})
        if len(classes) > 1 and min(classes.values()) < self.thresholds.imbalance:
            add("warning", self.text("eda_imbalance", share=min(classes.values())),
                rule="imbalance", classes=classes)
        return findings

    def next_events(self, event: Event) -> list[Event]:
        suggestion = self.suggest_configuration()
        if suggestion is None:
            return []
        data = {
            "target": suggestion.target,
            "task": suggestion.task,
            "features": list(suggestion.features),
        }
        return [Event(EDA_DONE, data, source=self.name)]

    def suggest_configuration(self) -> ModelingSuggestion | None:
        """Task, target and predictors without the columns the rules advise against."""
        custom = self.blackboard.results.get("modeling_suggestion")
        if custom is not None:
            return custom
        target = self.blackboard.results.get("profile", {}).get("target")
        if target is None:
            return None
        excluded = tuple(sorted({
            f.detail["column"] for f in self.blackboard.findings
            if f.agent == self.name and f.detail.get("rule") in EDA_EXCLUSION_RULES
        }))
        candidates = [str(c) for c in self.blackboard.datos.columns if c != target["name"]]
        features = [c for c in candidates if c not in excluded] or candidates
        return ModelingSuggestion(target["task"], target["name"], tuple(features), excluded)
