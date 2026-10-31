"""Agent base: reglas proactivas deterministas + bucle reactivo con LLM.

Ciclo al recibir un evento suscrito (:meth:`Agent.atender`):

1. :meth:`preparar` calcula lo indispensable sin LLM (barato y reproducible).
2. :meth:`proponer` aplica reglas propias y registra hallazgos (proactivo).
3. :meth:`reaccionar` deja que el LLM use herramientas e interprete (reactivo).
4. :meth:`siguientes` emite los eventos que activan a otros agentes.

Sin LLM, o si el proveedor falla (p. ej. límite del tier gratuito), los pasos
1, 2 y 4 siguen funcionando y el texto se arma con :meth:`resumen_sin_llm`.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

from ai_framework.utils.results import Hallazgo
from ai_framework.agents.events import PREGUNTA, EventBus, Event
from ai_framework.agents.tools.registry import Herramienta, ejecutar, serializar
from ai_framework.agents.llm.provider import ProveedorLLM
from ai_framework.agents.blackboard import Blackboard


class Agent(ABC):
    """Base común: cada subclase define sus reglas proactivas y herramientas."""

    nombre: str = "agente"
    rol: str = "Eres un asistente de análisis de datos."
    herramientas: tuple[Herramienta, ...] = ()
    escucha: frozenset[str] = frozenset()
    max_pasos: int = 6

    def __init__(self, pizarra: Blackboard, llm: ProveedorLLM | None = None) -> None:
        self.pizarra = pizarra
        self.llm = llm
        self.bus: EventBus | None = None

    def conectar(self, bus: EventBus) -> Agent:
        self.bus = bus
        for tipo in self.escucha:
            bus.suscribir(tipo, self.atender)
        return self

    # ---------------------------------------------------------- ciclo
    def atender(self, evento: Event) -> None:
        self.preparar(evento)
        nuevos = [h for h in self.proponer(evento) if self.pizarra.registrar(h)]
        texto = None
        if self.llm is not None:
            try:
                texto = self.reaccionar(evento)
            except Exception as error:
                self.pizarra.registrar(
                    Hallazgo(self.nombre, "aviso", f"LLM no disponible: {error}")
                )
        self.pizarra.anotar(self.nombre, texto or self.resumen_sin_llm(nuevos))
        if self.bus is not None:
            for siguiente in self.siguientes(evento):
                self.bus.emitir(siguiente)

    def preguntar(self, texto: str) -> str:
        """Atajo reactivo para una pregunta directa del usuario."""
        if self.llm is None:
            raise RuntimeError(f"{self.nombre} necesita un LLM para responder preguntas.")
        self.pizarra.anotar("usuario", texto)
        respuesta = self.reaccionar(Event(PREGUNTA, {"texto": texto}))
        self.pizarra.anotar(self.nombre, respuesta)
        return respuesta

    # ---------------------------------------------------------- ganchos
    def preparar(self, evento: Event) -> None:
        """Cálculos deterministas previos; por defecto no hace nada."""

    @abstractmethod
    def proponer(self, evento: Event) -> list[Hallazgo]:
        """Reglas proactivas deterministas sobre la pizarra."""

    def siguientes(self, evento: Event) -> list[Event]:
        return []

    def resumen_sin_llm(self, nuevos: list[Hallazgo]) -> str:
        if not nuevos:
            return f"{self.nombre}: sin observaciones nuevas."
        return "\n".join(f"[{h.nivel}] {h.mensaje}" for h in nuevos)

    # ---------------------------------------------------------- LLM
    def reaccionar(self, evento: Event) -> str:
        mensajes: list[dict[str, Any]] = [
            {"role": "system", "content": self.rol},
            {"role": "user", "content": self._instruccion(evento)},
        ]
        esquemas = [h.esquema() for h in self.herramientas]
        enviados = 0
        for paso in range(1, self.max_pasos + 1):
            # El historial se reenvía completo; en consola sólo se ve lo nuevo.
            for mensaje in mensajes[enviados:]:
                if mensaje["role"] != "assistant":
                    self._imprimir(f"ENVIADO paso {paso} ({mensaje['role']})", mensaje["content"])
            enviados = len(mensajes)
            try:
                respuesta = self.llm.completar(mensajes, esquemas)
            except Exception as error:
                self._imprimir(f"ERROR paso {paso}", str(error))
                raise
            self._imprimir(f"RECIBIDO paso {paso}", _describir_respuesta(respuesta))
            if not respuesta.llamadas:
                return respuesta.texto or ""
            mensajes.append(_mensaje_asistente(respuesta))
            for llamada in respuesta.llamadas:
                resultado = ejecutar(llamada, self.herramientas, self.pizarra)
                mensajes.append(
                    {"role": "tool", "tool_call_id": llamada.id, "content": serializar(resultado)}
                )
        return f"{self.nombre}: alcancé el límite de {self.max_pasos} pasos sin conclusión."

    def _instruccion(self, evento: Event) -> str:
        pedido = evento.datos.get("texto") or f"Ocurrió el evento '{evento.tipo}'."
        return f"{pedido}\n\nContexto (JSON):\n{self.pizarra.resumen_para_llm()}"

    def _imprimir(self, titulo: str, contenido: str) -> None:
        """Traza en la consola; un carácter no representable no debe tumbar al agente."""
        codificacion = getattr(sys.stdout, "encoding", None) or "utf-8"
        texto = f"\n--- [{self.nombre}] {titulo} ---\n{contenido}"
        print(texto.encode(codificacion, "replace").decode(codificacion), flush=True)


def _describir_respuesta(respuesta) -> str:
    llamadas = [f"-> {ll.nombre}({serializar(ll.argumentos)})" for ll in respuesta.llamadas]
    return "\n".join([*([respuesta.texto] if respuesta.texto else []), *llamadas])


def _mensaje_asistente(respuesta) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": respuesta.texto,
        "tool_calls": [
            {
                "id": llamada.id,
                "type": "function",
                "function": {"name": llamada.nombre, "arguments": serializar(llamada.argumentos)},
                **({"extra_content": llamada.extra} if llamada.extra else {}),
            }
            for llamada in respuesta.llamadas
        ],
    }
