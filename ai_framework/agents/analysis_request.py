"""Requests and outputs for configured model operations."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisRequest:
    specialty: str
    operation: str
    features: tuple[str, ...]
    options: dict[str, Any]


@dataclass(frozen=True)
class AnalysisResult:
    output: Any
    trace: dict[str, Any]
