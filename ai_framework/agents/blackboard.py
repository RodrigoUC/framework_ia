"""Estado compartido entre agentes (patrón pizarra / blackboard).

Hereda :class:`EDA`, igual que ``Supervisado`` y ``NoSupervisado``: los agentes
usan sus métodos de análisis sobre una sola copia del dataset. Al LLM sólo se
le envía :meth:`Blackboard.resumen_para_llm`, nunca las filas.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ai_framework.data.eda import EDA
from ai_framework.utils.results import Hallazgo


class Blackboard(EDA):
    # Únicos resultados que llegan al LLM; los demás pueden contener objetos o filas.
    CLAVES_CONTEXTO = ("perfil", "clustering")

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        ruta_datos=None,
        *,
        target: str | None = None,
        origen: str = "csv",
        **kwargs,
    ) -> None:
        super().__init__(dataframe=dataframe, ruta_datos=ruta_datos, **kwargs)
        if target is not None and target not in self.datos.columns:
            raise KeyError(f"La variable objetivo {target!r} no existe.")
        self.target = target  # None: sólo exploración, sin modelado sugerido
        self.origen = origen
        self.resultados: dict[str, Any] = {}
        self.hallazgos: list[Hallazgo] = []
        self.mensajes: list[tuple[str, str]] = []  # (autor, texto)

    def registrar(self, hallazgo: Hallazgo) -> bool:
        """Agrega el hallazgo si no estaba; devuelve si era nuevo."""
        if hallazgo in self.hallazgos:
            return False
        self.hallazgos.append(hallazgo)
        return True

    def anotar(self, autor: str, texto: str) -> None:
        self.mensajes.append((autor, texto))

    def resumen_para_llm(self, max_hallazgos: int = 20) -> str:
        """Contexto compacto: forma, columnas, perfil y hallazgos recientes."""
        contexto: dict[str, Any] = {
            "origen": self.origen,
            "filas": int(len(self.datos)),
            "columnas": [str(c) for c in self.datos.columns],
            "target": self.target,
        }
        for clave in self.CLAVES_CONTEXTO:
            if clave in self.resultados:
                contexto[clave] = self.resultados[clave]
        contexto["hallazgos"] = [
            {"agente": h.agente, "nivel": h.nivel, "mensaje": h.mensaje}
            for h in self.hallazgos[-max_hallazgos:]
        ]
        return json.dumps(contexto, ensure_ascii=False, default=str)
