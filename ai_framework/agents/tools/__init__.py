"""Tools that expose the framework to the agents."""

from __future__ import annotations

from ai_framework.agents.tools.classification_tools import CLASSIFICATION_TOOLS
from ai_framework.agents.tools.clustering_tools import CLUSTERING_TOOLS
from ai_framework.agents.tools.eda_tools import EDA_TOOLS
from ai_framework.agents.tools.regression_tools import REGRESSION_TOOLS
from ai_framework.agents.tools.registry import Tool, run_tool, tool

__all__ = [
    "CLASSIFICATION_TOOLS",
    "CLUSTERING_TOOLS",
    "EDA_TOOLS",
    "REGRESSION_TOOLS",
    "Tool",
    "run_tool",
    "tool",
]
