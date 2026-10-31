"""Contrato mínimo entre los agentes y cualquier modelo de lenguaje.

Los mensajes viajan en el formato de chat de OpenAI (``role``/``content``/
``tool_calls``), que es el que aceptan Gemini, Groq, OpenRouter y Ollama en sus
endpoints compatibles. Las respuestas se normalizan a :class:`RespuestaLLM`
para que los agentes no dependan del SDK de ningún proveedor.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LlamadaHerramienta:
    """Petición del modelo para ejecutar una herramienta registrada."""

    id: str
    nombre: str
    argumentos: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # opaco; se devuelve tal cual


@dataclass(frozen=True)
class RespuestaLLM:
    """Texto final o llamadas a herramientas; nunca ambos vacíos."""

    texto: str | None = None
    llamadas: tuple[LlamadaHerramienta, ...] = ()


class ProveedorLLM(Protocol):
    """Lo único que un agente necesita de un modelo de lenguaje."""

    def completar(
        self, mensajes: list[dict[str, Any]], herramientas: list[dict[str, Any]]
    ) -> RespuestaLLM: ...


class ProveedorFalso:
    """LLM guionizado para pruebas: devuelve respuestas en orden y registra todo.

    Permite probar el bucle de herramientas sin red ni API key.
    """

    def __init__(self, respuestas: Iterable[RespuestaLLM]) -> None:
        self._pendientes = list(respuestas)
        self.llamadas: list[dict[str, Any]] = []

    def completar(self, mensajes, herramientas):
        self.llamadas.append(
            {"mensajes": [dict(m) for m in mensajes], "herramientas": herramientas}
        )
        if not self._pendientes:
            raise RuntimeError("ProveedorFalso sin respuestas restantes.")
        return self._pendientes.pop(0)
