"""Proveedor para cualquier endpoint compatible con la API de chat de OpenAI.

El SDK ``openai`` se importa sólo al crear el cliente; si falta, se informa con
:class:`OptionalDependencyError`, igual que UMAP en el agrupamiento.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ai_framework.models.unsupervised import OptionalDependencyError
from ai_framework.utils.agents.agent_config import PROVEEDOR_SIN_LLM, PROVEEDORES_LLM
from ai_framework.agents.llm.provider import LlamadaHerramienta, RespuestaLLM

PISTAS_ESTADO = {
    400: "Puede ser la API key o un parámetro enviado.",
    401: "La API key no es válida.",
    403: "La key no tiene permiso: habilite la API en su proyecto, quite restricciones "
         "de la key o verifique que el servicio esté disponible en su país.",
    404: "El modelo no existe para este proveedor; revise AGENTES_MODELO.",
    429: "Se agotó la cuota gratuita o el límite por minuto; espere e intente de nuevo.",
}


class ErrorProveedor(RuntimeError):
    """Fallo HTTP del proveedor con el motivo devuelto y una pista para resolverlo."""

    def __init__(self, proveedor: str, modelo: str, estado: int, detalle: str) -> None:
        self.estado = estado
        pista = PISTAS_ESTADO.get(estado, "")
        super().__init__(
            f"{proveedor} ({modelo}) respondió {estado}: "
            f"{detalle.rstrip('. ') or 'sin detalle'}. {pista}".strip()
        )


class ProveedorOpenAICompat:
    """Envía el historial y los esquemas de herramientas; normaliza la respuesta."""

    def __init__(
        self,
        proveedor: str = "gemini",
        modelo: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperatura: float = 0.2,
        cliente: Any = None,
    ) -> None:
        if proveedor not in PROVEEDORES_LLM and base_url is None:
            raise ValueError(
                f"Proveedor desconocido {proveedor!r}; use {sorted(PROVEEDORES_LLM)} o base_url."
            )
        configuracion = PROVEEDORES_LLM.get(proveedor)
        self.proveedor = proveedor
        self.modelo = (
            modelo
            or os.environ.get("AGENTES_MODELO")
            or (configuracion.modelo_sugerido if configuracion else None)
        )
        if not self.modelo:
            raise ValueError("Indique el modelo o defina AGENTES_MODELO.")
        self.temperatura = temperatura
        if cliente is None:
            variable = configuracion.variable_clave if configuracion else None
            clave = api_key or (os.environ.get(variable) if variable else "ollama")
            if not clave:
                raise ValueError(f"Defina {variable} para usar {proveedor}.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Los agentes con LLM requieren 'openai'. Instale requirements.txt. "
                    f"Detalle técnico: {exc}"
                ) from exc
            cliente = OpenAI(base_url=base_url or configuracion.base_url, api_key=clave)
        self.cliente = cliente

    @classmethod
    def desde_entorno(cls) -> ProveedorOpenAICompat:
        """Lee ``AGENTES_PROVEEDOR`` (gemini por defecto) y ``AGENTES_MODELO``."""
        return cls(os.environ.get("AGENTES_PROVEEDOR", "gemini"))

    def completar(self, mensajes, herramientas):
        parametros: dict[str, Any] = {
            "model": self.modelo,
            "messages": mensajes,
            "temperature": self.temperatura,
        }
        if herramientas:
            parametros["tools"] = herramientas
        try:
            mensaje = self.cliente.chat.completions.create(**parametros).choices[0].message
        except Exception as error:
            estado = getattr(error, "status_code", None)
            if estado is None:
                raise
            raise ErrorProveedor(self.proveedor, self.modelo, estado, _detalle(error)) from error
        llamadas = tuple(
            LlamadaHerramienta(
                id=llamada.id,
                nombre=llamada.function.name,
                argumentos=_argumentos(llamada.function.arguments),
                # Gemini exige recibir de vuelta la thought_signature que viene aquí.
                extra=dict(getattr(llamada, "extra_content", None) or {}),
            )
            for llamada in (mensaje.tool_calls or ())
        )
        return RespuestaLLM(texto=mensaje.content, llamadas=llamadas)


def crear_proveedor(
    proveedor: str, modelo: str | None = None, api_key: str | None = None
) -> ProveedorOpenAICompat | None:
    """``None`` con :data:`PROVEEDOR_SIN_LLM`: los agentes trabajan sólo con reglas."""
    if proveedor == PROVEEDOR_SIN_LLM:
        return None
    return ProveedorOpenAICompat(proveedor, modelo or None, api_key=api_key or None)


def _detalle(error: Exception) -> str:
    """Extrae el mensaje del cuerpo JSON de error (formato OpenAI o Google)."""
    cuerpo = getattr(error, "body", None)
    if isinstance(cuerpo, list) and cuerpo:
        cuerpo = cuerpo[0]
    if isinstance(cuerpo, dict):
        interno = cuerpo.get("error", cuerpo)
        if isinstance(interno, dict) and interno.get("message"):
            return str(interno["message"])
    respuesta = getattr(error, "response", None)
    texto = getattr(respuesta, "text", "") or ""
    return texto[:300]


def _argumentos(crudo: str | None) -> dict[str, Any]:
    """JSON inválido se entrega a la herramienta como error explícito."""
    if not crudo:
        return {}
    try:
        valor = json.loads(crudo)
    except json.JSONDecodeError:
        return {"__json_invalido__": crudo}
    return valor if isinstance(valor, dict) else {"__json_invalido__": crudo}
