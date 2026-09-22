"""
SUPERVISADO — clase base del aprendizaje guiado (plantilla).

Conceptos (Notas_clase_4):
- Guiado.
- Normalmente la última columna es la target.
- La target puede ser categórica o numérica.
- Si el target es categórico → clasificación.
- Si el target es numérico → regresión (progresión).

Jerarquía:
    EDA
    └── Supervisado
           ├── Clasificacion
           └── Progresion
"""

from __future__ import annotations

import pandas as pd

from ..datos.eda import EDA


class Supervisado(EDA):
    """Base del aprendizaje supervisado.

    Atributos simples y públicos (``target``, ``features``, ``modelo``).
    Las subclases implementan explícitamente ``entrenar``, ``predecir`` y
    ``evaluar`` como parte del contrato de separación lógica del framework.
    """

    def __init__(
        self,
        dataframe=None,
        ruta_datos=None,
        *,
        target=None,
        features=None,
        **kwargs,
    ) -> None:
        super().__init__(dataframe=dataframe, ruta_datos=ruta_datos, **kwargs)
        self.target = target  # variable objetivo (None -> última columna)
        self.features = features  # predictoras (None -> todas menos target)
        self.modelo = None  # modelo entrenado (lo define la subclase)

    def _definir_target(self):
        """Devuelve el target, asumiendo la última columna si no se indicó."""
        if self.target is None:
            self.target = self.datos.columns[-1]
        return self.target

    def _tipo_target(self):
        """'clasificacion' si el target es categórico; 'progresion' si es numérico."""
        self._definir_target()
        serie = self.datos[self.target]
        if pd.api.types.is_numeric_dtype(serie):
            return "progresion"
        return "clasificacion"

    def _definir_features(self):
        """Devuelve las features (todas las columnas menos el target)."""
        self._definir_target()
        if self.features is None:
            self.features = [
                columna for columna in self.datos.columns if columna != self.target
            ]
        return self.features

    def entrenar(self, *args, **kwargs):
        """TODO: entrenar el modelo supervisado."""
        raise NotImplementedError("Debe implementar 'entrenar' en la subclase.")

    def predecir(self, X=None):
        """TODO: generar predicciones con el modelo entrenado."""
        raise NotImplementedError("Debe implementar 'predecir' en la subclase.")

    def evaluar(self):
        """TODO: evaluar el desempeño del modelo."""
        raise NotImplementedError("Debe implementar 'evaluar' en la subclase.")

    def __str__(self) -> str:
        self._definir_target()
        return (
            "Clase Supervisado\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
