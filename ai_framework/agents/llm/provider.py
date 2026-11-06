"""Minimal contract between the agents and any language model.

Messages use the OpenAI chat format (``role``/``content``/``tool_calls``),
which Gemini, Groq, OpenRouter and Ollama accept on their compatible
endpoints. Responses are normalized to :class:`LLMResponse` so the agents do
not depend on any provider's SDK.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

# Argument key used when the model sends arguments that are not valid JSON.
INVALID_JSON_KEY = "__invalid_json__"


@dataclass(frozen=True)
class ToolCall:
    """Request from the model to run a registered tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # opaque; sent back unchanged


@dataclass(frozen=True)
class LLMResponse:
    """Final text or tool calls; never both empty."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class LLMProvider(Protocol):
    """The only thing an agent needs from a language model."""

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


class FakeLLMProvider:
    """Scripted LLM for tests: returns responses in order and records every call.

    Lets the tool loop be tested without network access or an API key.
    """

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._pending = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._pending:
            raise RuntimeError("FakeLLMProvider has no responses left.")
        return self._pending.pop(0)
