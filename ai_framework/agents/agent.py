"""Base agent: deterministic proactive rules + reactive loop with an LLM.

Cycle when a subscribed event arrives (:meth:`Agent.handle`):

0. :meth:`accepts` filters events that do not concern the agent.
1. :meth:`prepare` computes what is needed without the LLM (cheap, reproducible).
2. :meth:`propose` applies the agent's own rules and records findings (proactive).
3. :meth:`react` lets the LLM use tools and interpret the results (reactive).
4. :meth:`next_events` emits the events that trigger other agents.

Without an LLM, or if the provider fails (e.g. free-tier rate limit), steps
1, 2 and 4 keep working and the text comes from :meth:`summary_without_llm`.

Every step is reported through :meth:`report`, which records the update on the
blackboard and passes it to ``on_progress`` so the UI or the CLI can show it
while the agent works. Texts follow ``language`` (see :mod:`messages`).
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ai_framework.utils.results import Finding
from ai_framework.agents.events import QUESTION, Event, EventBus
from ai_framework.agents.messages import language_instruction, message, normalize_language
from ai_framework.agents.tools.registry import Tool, run_tool, serialize
from ai_framework.agents.llm.provider import LLMProvider
from ai_framework.agents.blackboard import Blackboard

ProgressCallback = Callable[[str], None]


class Agent(ABC):
    """Common base: each subclass defines its proactive rules and tools."""

    name: str = "agent"
    role: str = "You are a data analysis assistant."
    tools: tuple[Tool, ...] = ()
    listens_to: frozenset[str] = frozenset()
    max_steps: int = 6

    def __init__(
        self,
        blackboard: Blackboard,
        llm: LLMProvider | None = None,
        *,
        language: str = "es",
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.blackboard = blackboard
        self.llm = llm
        self.language = normalize_language(language)
        self.on_progress = on_progress
        self.bus: EventBus | None = None

    def connect(self, bus: EventBus) -> Agent:
        self.bus = bus
        for event_type in self.listens_to:
            bus.subscribe(event_type, self.handle)
        return self

    # ---------------------------------------------------------- language and progress
    def text(self, key: str, **values) -> str:
        """User-facing text in the agent's language."""
        return message(self.language, key, **values)

    @property
    def display_name(self) -> str:
        return self.text(f"agent_{self.name}")

    def report(self, key: str, **values) -> None:
        """Records a progress update and forwards it to whoever is watching."""
        update = self.text(key, **values)
        self.blackboard.progress.append((self.name, update))
        if self.on_progress is not None:
            self.on_progress(update)

    # ---------------------------------------------------------- cycle
    def handle(self, event: Event) -> None:
        if not self.accepts(event):
            return
        self.report("progress_start", agent=self.display_name)
        self.prepare(event)
        new_findings = [f for f in self.propose(event) if self.blackboard.add_finding(f)]
        levels = [f.level for f in new_findings]
        self.report(
            "progress_rules",
            total=len(new_findings),
            critical=levels.count("critical"),
            warnings=levels.count("warning"),
        )
        text = None
        if self.llm is not None:
            try:
                text = self.react(event)
                self.report("progress_llm_done")
            except Exception as error:
                self.report("progress_llm_failed")
                self.blackboard.add_finding(
                    Finding(self.name, "warning", self.text("llm_unavailable", error=error))
                )
        self.blackboard.add_message(self.name, text or self.summary_without_llm(new_findings))
        if self.bus is not None:
            for next_event in self.next_events(event):
                self.report("progress_next", event=next_event.type)
                self.bus.emit(next_event)
        self.report("progress_done", agent=self.display_name)

    def ask(self, text: str) -> str:
        """Reactive shortcut for a direct question from the user."""
        if self.llm is None:
            raise RuntimeError(self.text("needs_llm", agent=self.display_name))
        self.blackboard.add_message("user", text)
        answer = self.react(Event(QUESTION, {"text": text}))
        self.report("progress_llm_done")
        self.blackboard.add_message(self.name, answer)
        return answer

    # ---------------------------------------------------------- hooks
    def accepts(self, event: Event) -> bool:
        """Whether a subscribed event concerns this agent; all of them by default."""
        return True

    def prepare(self, event: Event) -> None:
        """Deterministic computations before the rules; does nothing by default."""

    @abstractmethod
    def propose(self, event: Event) -> list[Finding]:
        """Deterministic proactive rules over the blackboard."""

    def next_events(self, event: Event) -> list[Event]:
        return []

    def summary_without_llm(self, new_findings: list[Finding]) -> str:
        if not new_findings:
            return self.text("no_new_observations", agent=self.display_name)
        return "\n".join(f"[{f.level}] {f.message}" for f in new_findings)

    # ---------------------------------------------------------- LLM
    def system_prompt(self) -> str:
        return f"{self.role}\n\n{language_instruction(self.language)}"

    def react(self, event: Event) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self._instruction(event)},
        ]
        schemas = [t.schema() for t in self.tools]
        sent = 0
        for step in range(1, self.max_steps + 1):
            self.report("progress_llm_step", step=step)
            # The whole history is resent; the console only shows what is new.
            for item in messages[sent:]:
                if item["role"] != "assistant":
                    self._log(f"SENT step {step} ({item['role']})", item["content"])
            sent = len(messages)
            try:
                response = self.llm.complete(messages, schemas)
            except Exception as error:
                self._log(f"ERROR step {step}", str(error))
                raise
            self._log(f"RECEIVED step {step}", _describe_response(response))
            if not response.tool_calls:
                return response.text or ""
            messages.append(_assistant_message(response))
            for call in response.tool_calls:
                self.report("progress_llm_tool", tool=call.name)
                result = run_tool(call, self.tools, self.blackboard)
                if "error" in result:
                    self.report("progress_llm_tool_error", tool=call.name)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": serialize(result)}
                )
        return self.text("step_limit", agent=self.display_name, steps=self.max_steps)

    def _instruction(self, event: Event) -> str:
        request = event.data.get("text") or (
            f"The event '{event.type}' happened. Analyze the context below and report "
            "what matters most to the user."
        )
        return f"{request}\n\nContext (JSON):\n{self.blackboard.llm_context()}"

    def _log(self, title: str, content: str) -> None:
        """Console trace; a character the console cannot encode must not stop the agent."""
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        text = f"\n--- [{self.name}] {title} ---\n{content}"
        print(text.encode(encoding, "replace").decode(encoding), flush=True)


def _describe_response(response) -> str:
    calls = [f"-> {call.name}({serialize(call.arguments)})" for call in response.tool_calls]
    return "\n".join([*([response.text] if response.text else []), *calls])


def _assistant_message(response) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.text,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": serialize(call.arguments)},
                **({"extra_content": call.extra} if call.extra else {}),
            }
            for call in response.tool_calls
        ],
    }
