"""State shared by the agents (blackboard pattern).

Inherits :class:`EDA`, like ``Supervised`` and ``Unsupervised``: agents use its
analysis methods on a single copy of the dataset. Only
:meth:`Blackboard.llm_context` is sent to the LLM, never the rows.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ai_framework.data.eda import EDA
from ai_framework.utils.results import Finding


class Blackboard(EDA):
    # Only results that reach the LLM; the others may hold objects or rows.
    CONTEXT_KEYS = ("profile", "clustering", "regression", "classification")

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        ruta_datos=None,
        *,
        target: str | None = None,
        source: str = "csv",
        **kwargs,
    ) -> None:
        super().__init__(dataframe=dataframe, ruta_datos=ruta_datos, **kwargs)
        if target is not None and target not in self.datos.columns:
            raise KeyError(f"La variable objetivo {target!r} no existe.")
        self.target = target  # None: exploration only, no modeling suggestion
        self.source = source
        self.results: dict[str, Any] = {}
        self.findings: list[Finding] = []
        self.messages: list[tuple[str, str]] = []  # (author, text)
        self.progress: list[tuple[str, str]] = []  # (agent, update)

    def add_finding(self, finding: Finding) -> bool:
        """Adds the finding if it was not there yet; returns whether it was new."""
        if finding in self.findings:
            return False
        self.findings.append(finding)
        return True

    def add_message(self, author: str, text: str) -> None:
        self.messages.append((author, text))

    def llm_context(self, max_findings: int = 20) -> str:
        """Compact context: shape, columns, profile and recent findings."""
        context: dict[str, Any] = {
            "source": self.source,
            "rows": int(len(self.datos)),
            "columns": [str(c) for c in self.datos.columns],
            "target": self.target,
        }
        for key in self.CONTEXT_KEYS:
            if key in self.results:
                context[key] = self.results[key]
        context["findings"] = [
            {"agent": f.agent, "level": f.level, "message": f.message}
            for f in self.findings[-max_findings:]
        ]
        return json.dumps(context, ensure_ascii=False, default=str)
