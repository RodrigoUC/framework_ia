"""Tools the LLM can call, and their JSON schema.

A tool is a function ``f(blackboard, **arguments) -> dict`` that calls the
framework and returns a serializable summary. The schema is derived from the
annotations, so the LLM can only request what the signature allows.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ai_framework.agents.llm.provider import INVALID_JSON_KEY, ToolCall

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., dict[str, Any]]
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    description: str, **parameter_descriptions: str
) -> Callable[[Callable[..., dict[str, Any]]], Tool]:
    """Decorates ``f(blackboard, ...)``; the first parameter is never exposed to the LLM."""

    def decorate(function):
        signature = inspect.signature(function)
        hints = typing.get_type_hints(function)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in list(signature.parameters.items())[1:]:
            schema, optional = _type_schema(hints.get(name, str))
            if name in parameter_descriptions:
                schema["description"] = parameter_descriptions[name]
            properties[name] = schema
            if parameter.default is inspect.Parameter.empty and not optional:
                required.append(name)
        parameters = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return Tool(function.__name__, description, function, parameters)

    return decorate


def _type_schema(annotation) -> tuple[dict[str, Any], bool]:
    """Translates ``str``, ``int``, ``float``, ``bool``, ``list[T]`` and ``T | None``."""
    arguments = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        real = [a for a in arguments if a is not type(None)]
        if len(real) != 1:
            raise TypeError(f"Unsupported union in tool: {annotation}")
        return _type_schema(real[0])[0], True
    if typing.get_origin(annotation) is list:
        items = _type_schema(arguments[0] if arguments else str)[0]
        return {"type": "array", "items": items}, False
    if annotation not in _JSON_TYPES:
        raise TypeError(f"Unsupported type in tool: {annotation}")
    return {"type": _JSON_TYPES[annotation]}, False


def run_tool(call: ToolCall, tools: Iterable[Tool], blackboard) -> dict[str, Any]:
    """Runs the call; any failure goes back to the LLM as ``{"error": ...}``."""
    available = {t.name: t for t in tools}
    chosen = available.get(call.name)
    if chosen is None:
        return {"error": f"Herramienta desconocida {call.name!r}; use {sorted(available)}."}
    if INVALID_JSON_KEY in call.arguments:
        return {"error": "Los argumentos no son JSON válido."}
    unknown = set(call.arguments) - set(chosen.parameters["properties"])
    if unknown:
        return {"error": f"Argumentos no admitidos: {sorted(unknown)}."}
    try:
        return chosen.function(blackboard, **call.arguments)
    except Exception as error:  # the LLM receives the failure and can correct itself
        return {"error": f"{type(error).__name__}: {error}"}


def serialize(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
