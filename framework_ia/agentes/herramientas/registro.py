"""Herramientas que el LLM puede invocar y su esquema JSON.

Una herramienta es una función ``f(pizarra, **argumentos) -> dict`` que llama al
framework y devuelve un resumen serializable. El esquema se deriva de las
anotaciones, así el LLM sólo puede pedir lo que la firma admite.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..llm.proveedor import LlamadaHerramienta

_TIPOS_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Herramienta:
    nombre: str
    descripcion: str
    funcion: Callable[..., dict[str, Any]]
    parametros: dict[str, Any]

    def esquema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.nombre,
                "description": self.descripcion,
                "parameters": self.parametros,
            },
        }


def herramienta(
    descripcion: str, **descripciones_parametros: str
) -> Callable[[Callable[..., dict[str, Any]]], Herramienta]:
    """Decora ``f(pizarra, ...)``; el primer parámetro nunca se expone al LLM."""

    def decorar(funcion):
        firma = inspect.signature(funcion)
        pistas = typing.get_type_hints(funcion)
        propiedades: dict[str, Any] = {}
        requeridos: list[str] = []
        for nombre, parametro in list(firma.parameters.items())[1:]:
            esquema, opcional = _esquema_tipo(pistas.get(nombre, str))
            if nombre in descripciones_parametros:
                esquema["description"] = descripciones_parametros[nombre]
            propiedades[nombre] = esquema
            if parametro.default is inspect.Parameter.empty and not opcional:
                requeridos.append(nombre)
        parametros = {"type": "object", "properties": propiedades}
        if requeridos:
            parametros["required"] = requeridos
        return Herramienta(funcion.__name__, descripcion, funcion, parametros)

    return decorar


def _esquema_tipo(tipo) -> tuple[dict[str, Any], bool]:
    """Traduce ``str``, ``int``, ``float``, ``bool``, ``list[T]`` y ``T | None``."""
    argumentos = typing.get_args(tipo)
    if typing.get_origin(tipo) in (typing.Union, types.UnionType):
        reales = [a for a in argumentos if a is not type(None)]
        if len(reales) != 1:
            raise TypeError(f"Unión no soportada en herramienta: {tipo}")
        return _esquema_tipo(reales[0])[0], True
    if typing.get_origin(tipo) is list:
        interno = _esquema_tipo(argumentos[0] if argumentos else str)[0]
        return {"type": "array", "items": interno}, False
    if tipo not in _TIPOS_JSON:
        raise TypeError(f"Tipo no soportado en herramienta: {tipo}")
    return {"type": _TIPOS_JSON[tipo]}, False


def ejecutar(
    llamada: LlamadaHerramienta, herramientas: Iterable[Herramienta], pizarra
) -> dict[str, Any]:
    """Ejecuta la llamada; cualquier fallo vuelve como ``{"error": ...}`` al LLM."""
    disponibles = {h.nombre: h for h in herramientas}
    elegida = disponibles.get(llamada.nombre)
    if elegida is None:
        return {"error": f"Herramienta desconocida {llamada.nombre!r}; use {sorted(disponibles)}."}
    if "__json_invalido__" in llamada.argumentos:
        return {"error": "Los argumentos no son JSON válido."}
    desconocidos = set(llamada.argumentos) - set(elegida.parametros["properties"])
    if desconocidos:
        return {"error": f"Argumentos no admitidos: {sorted(desconocidos)}."}
    try:
        return elegida.funcion(pizarra, **llamada.argumentos)
    except Exception as error:  # el LLM recibe el fallo y puede corregirse
        return {"error": f"{type(error).__name__}: {error}"}


def serializar(resultado: dict[str, Any]) -> str:
    return json.dumps(resultado, ensure_ascii=False, default=str)
