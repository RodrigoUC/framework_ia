"""Bus de eventos síncrono que encadena agentes.

Los eventos emitidos mientras se atiende otro se encolan y se procesan en
orden (FIFO), así un agente puede disparar al siguiente sin recursión.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DATASET_CARGADO = "dataset_cargado"
EDA_LISTO = "eda_listo"
CLUSTERING_SOLICITADO = "clustering_solicitado"
CLUSTERING_LISTO = "clustering_listo"
MODELO_LISTO = "modelo_listo"
PREGUNTA = "pregunta"


@dataclass(frozen=True)
class Event:
    tipo: str
    datos: dict[str, Any] = field(default_factory=dict)
    origen: str = "usuario"


Manejador = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._suscriptores: dict[str, list[Manejador]] = defaultdict(list)
        self._cola: deque[Event] = deque()
        self._procesando = False
        self.historial: list[Event] = []

    def suscribir(self, tipo: str, manejador: Manejador) -> None:
        self._suscriptores[tipo].append(manejador)

    def emitir(self, evento: Event) -> None:
        self._cola.append(evento)
        if self._procesando:
            return
        self._procesando = True
        try:
            while self._cola:
                actual = self._cola.popleft()
                self.historial.append(actual)
                for manejador in list(self._suscriptores[actual.tipo]):
                    manejador(actual)
        finally:
            self._procesando = False
            self._cola.clear()
