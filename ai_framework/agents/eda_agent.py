"""Agente de análisis exploratorio construido sobre :class:`Agent`."""

from __future__ import annotations

from ai_framework.utils.results import Hallazgo, SugerenciaModelado
from ai_framework.utils.agents.agent_config import REGLAS_EXCLUSION_EDA, UmbralesEDA
from ai_framework.agents.agent import Agent
from ai_framework.agents.events import DATASET_CARGADO, EDA_LISTO, Event
from ai_framework.agents.tools.eda_tools import (
    HERRAMIENTAS_EXPLORACION,
    parece_identificador,
    perfil_dataset,
)


class EDAAgent(Agent):
    """Perfila el dataset al cargarse, avisa problemas y habilita el modelado."""

    nombre = "eda"
    rol = (
        "Eres el agente de análisis exploratorio de un framework de aprendizaje "
        "automático. Usa sólo las herramientas para obtener cifras; no inventes "
        "valores. Explica en español, en pocos párrafos, la calidad del dataset, "
        "riesgos para el modelado (nulos, atípicos, colinealidad, fuga de "
        "información, desbalance) y el siguiente paso recomendado. Nunca sugieras "
        "usar el conjunto de prueba para tomar decisiones."
    )
    herramientas = HERRAMIENTAS_EXPLORACION
    escucha = frozenset({DATASET_CARGADO})

    def __init__(self, pizarra, llm=None, *, umbrales: UmbralesEDA | None = None) -> None:
        super().__init__(pizarra, llm)
        self.umbrales = umbrales or UmbralesEDA()

    def preparar(self, evento: Event) -> None:
        perfil_dataset.funcion(self.pizarra)

    def proponer(self, evento: Event) -> list[Hallazgo]:
        perfil = self.pizarra.resultados["perfil"]
        calidad = perfil["calidad"]
        hallazgos: list[Hallazgo] = []

        def registrar(nivel, mensaje, **detalle):
            hallazgos.append(Hallazgo(self.nombre, nivel, mensaje, detalle))

        if calidad["duplicados"]:
            registrar("aviso", f"Hay {calidad['duplicados']} filas duplicadas.",
                      regla="duplicados", duplicados=calidad["duplicados"])
        for columna in perfil["columnas"]:
            nombre = columna["columna"]
            if nombre == self.pizarra.target:
                continue
            if columna["porcentaje_nulos"] > self.umbrales.nulos:
                registrar("aviso", f"'{nombre}' tiene {columna['porcentaje_nulos']}% de nulos.",
                          regla="nulos", columna=nombre)
            if columna["valores_unicos"] <= 1:
                registrar("aviso", f"'{nombre}' es constante y no aporta información.",
                          regla="constante", columna=nombre)
            elif parece_identificador(
                self.pizarra.datos[nombre], self.umbrales.min_filas_identificador
            ):
                registrar("aviso", f"'{nombre}' parece un identificador; conviene excluirla.",
                          regla="identificador", columna=nombre)

        objetivo = perfil.get("target")
        if objetivo is None:
            registrar("info", "No hay variable objetivo; defínala para habilitar el modelado.")
            return hallazgos
        for columna, r in objetivo.get("correlaciones", {}).items():
            if abs(r) >= self.umbrales.fuga:
                registrar("critico",
                          f"'{columna}' tiene r={r} con el target: posible fuga de información.",
                          regla="fuga", columna=columna, r=r)
        clases = objetivo.get("clases", {})
        if len(clases) > 1 and min(clases.values()) < self.umbrales.desbalance:
            registrar("aviso",
                      f"Target desbalanceado: la clase minoritaria es {min(clases.values()):.1%}.",
                      regla="desbalance", clases=clases)
        return hallazgos

    def siguientes(self, evento: Event) -> list[Event]:
        objetivo = self.pizarra.resultados["perfil"].get("target")
        if objetivo is None:
            return []
        return [
            Event(
                EDA_LISTO,
                {"target": objetivo["nombre"], "tarea": objetivo["tarea"]},
                origen=self.nombre,
            )
        ]

    def sugerir_configuracion(self) -> SugerenciaModelado | None:
        """Tarea, target y predictoras sin las columnas que las reglas desaconsejan."""
        objetivo = self.pizarra.resultados.get("perfil", {}).get("target")
        if objetivo is None:
            return None
        excluidas = tuple(sorted({
            h.detalle["columna"] for h in self.pizarra.hallazgos
            if h.agente == self.nombre and h.detalle.get("regla") in REGLAS_EXCLUSION_EDA
        }))
        candidatas = [str(c) for c in self.pizarra.datos.columns if c != objetivo["nombre"]]
        features = [c for c in candidatas if c not in excluidas] or candidatas
        return SugerenciaModelado(objetivo["tarea"], objetivo["nombre"], tuple(features), excluidas)
