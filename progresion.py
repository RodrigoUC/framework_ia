"""
PROGRESIÓN — Aprendizaje supervisado con target numérico (plantilla).

Cuando el target es numérico (continuo o discreto), la investigación es de
regresión/progresión: se busca calcular un valor exacto.
Hereda de :class:`Supervisado`.

Métodos a desarrollar (vacíos por ahora):
- RLS → Regresión Lineal Simple
- RLM → Regresión Lineal Múltiple
- RL  → Regresión Logística

Jerarquía:
    EDA
    └── Supervisado
           └── Progresion
"""

from __future__ import annotations

from supervisado import Supervisado


class Progresion(Supervisado):
    """Progresión (regresión) supervisada con métodos pendientes de implementar."""

    def __init__(
        self,
        dataframe=None,
        ruta_datos=None,
        *,
        target=None,
        features=None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataframe=dataframe,
            ruta_datos=ruta_datos,
            target=target,
            features=features,
            **kwargs,
        )
        self.r2 = None  # coeficiente de determinación (evaluación)

    def RLS(self, *args, **kwargs):
        """Regresión Lineal Simple.

        TODO: implementar la regresión con una única variable predictora.
        """
        raise NotImplementedError("Debe implementar el método RLS.")

    def RLM(self, *args, **kwargs):
        """Regresión Lineal Múltiple.

        TODO: implementar la regresión con varias variables predictoras.
        """
        raise NotImplementedError("Debe implementar el método RLM.")

    def RL(self, *args, **kwargs):
        """Regresión Logística.

        TODO: implementar la regresión logística (modelo de clasificación).
        """
        raise NotImplementedError("Debe implementar el método RL.")

    def __str__(self) -> str:
        self._definir_target()
        base = (
            "Clase Progresion\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
        if self.r2 is not None:
            base += f"\nR²: {self.r2:.4f}"
        return base