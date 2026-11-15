"""Provider for any endpoint compatible with the OpenAI chat API.

The ``openai`` SDK is imported only when the client is created; if it is
missing, :class:`OptionalDependencyError` is raised, as UMAP does in clustering.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ai_framework.models.unsupervised import OptionalDependencyError
from ai_framework.utils.agents.agent_config import LLM_PROVIDERS, RULES_ONLY
from ai_framework.agents.llm.provider import INVALID_JSON_KEY, LLMResponse, ToolCall

STATUS_HINTS = {
    400: "Puede ser la API key o un parámetro enviado.",
    401: "La API key no es válida.",
    403: "La key no tiene permiso: habilite la API en su proyecto, quite restricciones "
         "de la key o verifique que el servicio esté disponible en su país.",
    404: "El modelo no existe para este proveedor; revise AGENTES_MODELO.",
    429: "Se agotó la cuota gratuita o el límite por minuto; espere e intente de nuevo.",
}


class ProviderError(RuntimeError):
    """HTTP failure from the provider, with the returned reason and a hint to fix it."""

    def __init__(self, provider: str, model: str, status: int, detail: str) -> None:
        self.status = status
        hint = STATUS_HINTS.get(status, "")
        super().__init__(
            f"{provider} ({model}) respondió {status}: "
            f"{detail.rstrip('. ') or 'sin detalle'}. {hint}".strip()
        )


class OpenAICompatibleProvider:
    """Sends the history and the tool schemas; normalizes the response."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        client: Any = None,
    ) -> None:
        if provider not in LLM_PROVIDERS and base_url is None:
            raise ValueError(
                f"Proveedor desconocido {provider!r}; use {sorted(LLM_PROVIDERS)} o base_url."
            )
        config = LLM_PROVIDERS.get(provider)
        self.provider = provider
        self.model = (
            model
            or os.environ.get("AGENTES_MODELO")
            or (config.suggested_model if config else None)
        )
        if not self.model:
            raise ValueError("Indique el modelo o defina AGENTES_MODELO.")
        self.temperature = temperature
        if client is None:
            variable = config.key_variable if config else None
            key = api_key or (os.environ.get(variable) if variable else "ollama")
            if not key:
                raise ValueError(f"Defina {variable} para usar {provider}.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Los agentes con LLM requieren 'openai'. Instale requirements.txt. "
                    f"Detalle técnico: {exc}"
                ) from exc
            client = OpenAI(base_url=base_url or config.base_url, api_key=key)
        self.client = client

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        """Reads ``AGENTES_PROVEEDOR`` (gemini by default) and ``AGENTES_MODELO``."""
        return cls(os.environ.get("AGENTES_PROVEEDOR", "gemini"))

    def complete(self, messages, tools):
        parameters: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            parameters["tools"] = tools
        try:
            message = self.client.chat.completions.create(**parameters).choices[0].message
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status is None:
                raise
            raise ProviderError(self.provider, self.model, status, _error_detail(error)) from error
        tool_calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
                # Gemini requires the thought_signature that comes here to be sent back.
                extra=dict(getattr(call, "extra_content", None) or {}),
            )
            for call in (message.tool_calls or ())
        )
        return LLMResponse(text=message.content, tool_calls=tool_calls)


def create_provider(
    provider: str, model: str | None = None, api_key: str | None = None
) -> OpenAICompatibleProvider | None:
    """``None`` for :data:`RULES_ONLY`: the agents then work with their rules only."""
    if provider == RULES_ONLY:
        return None
    return OpenAICompatibleProvider(provider, model or None, api_key=api_key or None)


def _error_detail(error: Exception) -> str:
    """Extracts the message from the JSON error body (OpenAI or Google format)."""
    body = getattr(error, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    response = getattr(error, "response", None)
    text = getattr(response, "text", "") or ""
    return text[:300]


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Invalid JSON is handed to the tool runner as an explicit error."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {INVALID_JSON_KEY: raw}
    return value if isinstance(value, dict) else {INVALID_JSON_KEY: raw}
