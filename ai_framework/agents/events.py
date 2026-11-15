"""Synchronous event bus that chains agents together.

Events emitted while another one is being handled are queued and processed in
order (FIFO), so an agent can trigger the next one without recursion.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DATASET_LOADED = "dataset_loaded"
EDA_DONE = "eda_done"
CLUSTERING_REQUESTED = "clustering_requested"
CLUSTERING_DONE = "clustering_done"
REGRESSION_REQUESTED = "regression_requested"
REGRESSION_DONE = "regression_done"
CLASSIFICATION_REQUESTED = "classification_requested"
CLASSIFICATION_DONE = "classification_done"
MODEL_DONE = "model_done"
QUESTION = "question"


@dataclass(frozen=True)
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "user"


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._queue: deque[Event] = deque()
        self._processing = False
        self.history: list[Event] = []

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        self._queue.append(event)
        if self._processing:
            return
        self._processing = True
        try:
            while self._queue:
                current = self._queue.popleft()
                self.history.append(current)
                for handler in list(self._subscribers[current.type]):
                    handler(current)
        finally:
            self._processing = False
            self._queue.clear()
