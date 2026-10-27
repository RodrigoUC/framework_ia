"""
SUPERVISADO — clase base del aprendizaje guiado.

Conceptos:
- Aprendizaje guiado con variable objetivo (``target``) y predictoras (``features``).
- Normalmente la última columna es la ``target`` si no se especifica explícitamente.
- Si el target es categórico/discreto → clasificación (:class:`Clasificacion`).
- Si el target es numérico continuo → regresión (:class:`Regresion`).

Jerarquía:
    DataFrame
    └── EDA
        └── Supervisado
               ├── Clasificacion
               └── Regresion
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...datos.eda import EDA
from ...datos.particion import dividir_indices, validar_particion
from ...resultados import ResultadoParticion


class Supervisado(EDA):
    """Base común del aprendizaje supervisado.

    Proporciona validación estructural de ``target`` y ``features``, manejo de
    particiones sin fuga de datos, transformación de características y contrato
    completo de ``entrenar``, ``predecir``, ``evaluar``, ``experimentar`` y
    ``previsualizar_particion``.
    """

    _mensaje_sin_entrenar = "Entrene primero un modelo supervisado."

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
        self.modelo = None  # modelo entrenado
        self.resultado: Any = None
        self.experimento: Any = None
        self._preparados: Any = None

    def _reiniciar_resultados(self) -> None:
        """Limpia artefactos previos antes de iniciar un nuevo entrenamiento."""
        self.resultado = None
        self.experimento = None
        self.modelo = None
        self._preparados = None

    def _definir_target(self):
        """Devuelve el target, asumiendo la última columna si no se indicó."""
        if self.target is None:
            self.target = self.datos.columns[-1]
        return self.target

    def _tipo_target(self):
        """Indica clasificación si el target es categórico o regresión si es numérico."""
        self._definir_target()
        serie = self.datos[self.target]
        if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(
            serie
        ):
            return "regresion"
        return "clasificacion"

    def _definir_features(self):
        """Devuelve las features (todas las columnas menos el target)."""
        self._definir_target()
        if self.features is None:
            self.features = [
                columna for columna in self.datos.columns if columna != self.target
            ]
        return self.features

    def _validar_estructura_base(
        self,
        *,
        mensaje_indices: str = "El dataset necesita índices únicos.",
    ) -> pd.DataFrame:
        """Valida la integridad tabular común a clasificación y regresión."""
        if self.datos.empty:
            raise ValueError("No hay datos para entrenar.")
        if not self.datos.index.is_unique:
            raise ValueError(mensaje_indices)
        if not self.datos.columns.is_unique:
            raise ValueError("El dataset necesita nombres de columna únicos.")
        self._definir_target()
        if self.target not in self.datos:
            raise KeyError(f"La variable objetivo '{self.target}' no existe.")
        self.features = list(self._definir_features())
        if not self.features:
            raise ValueError("Debe seleccionar al menos una variable predictora.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Las variables predictoras no deben repetirse.")
        if self.target in self.features:
            raise ValueError("El target no debe incluirse entre las features.")
        faltantes = [c for c in self.features if c not in self.datos]
        if faltantes:
            raise KeyError(f"Columnas inexistentes: {faltantes}")
        return self.datos.loc[:, self.features + [self.target]].copy()

    def _indices(
        self,
        base: pd.DataFrame,
        *,
        particion: ResultadoParticion | None,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = False,
    ) -> tuple[pd.Index, pd.Index, pd.Index]:
        """Obtiene los índices de train, test y validación opcional sin fuga de datos."""
        if particion is not None:
            validar_particion(particion, self.datos)
            indices = [
                subset.index[subset.index.isin(base.index)]
                if subset is not None
                else pd.Index([])
                for subset in (particion.train, particion.test, particion.validacion)
            ]
            if any(indice.empty for indice in indices[:2]):
                raise ValueError("Train o test quedó vacío al quitar objetivos nulos.")
            if particion.validacion is not None and indices[2].empty:
                raise ValueError("Validación quedó vacía al quitar objetivos nulos.")
            return tuple(indices)
        train, test = dividir_indices(
            base,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return train, test, pd.Index([])

    def _datos_supervisados(self) -> pd.DataFrame:
        """Obtiene el DataFrame base validado según la tarea supervisada."""
        return self._validar_estructura_base()

    def _delegado_supervisado(self, algoritmo: str | None = None) -> "Supervisado":
        """Crea el especialista adecuado cuando se usa ``Supervisado`` directamente."""
        from .clasificacion import Clasificacion
        from .regresion import Regresion

        if algoritmo is not None:
            alg_norm = str(algoritmo).strip().lower()
            if alg_norm in {
                "knn",
                "XGBoost".lower(),
                "xgb",
                "adaboost",
                "ada",
                "nr",
                "nb",
                "naive_bayes",
                "rl",
                "regresion_logistica",
                "logistic_regression",
            }:
                return Clasificacion(
                    dataframe=self.datos, target=self.target, features=self.features
                )
            if alg_norm in {
                "rls",
                "rlm",
                "lineal_simple",
                "lineal_multiple",
                "lasso",
                "lassocv",
                "lasso_cv",
                "ridge",
                "ridgecv",
                "ridge_cv",
                "svr",
            }:
                return Regresion(
                    dataframe=self.datos, target=self.target, features=self.features
                )
        clase = Regresion if self._tipo_target() == "regresion" else Clasificacion
        return clase(dataframe=self.datos, target=self.target, features=self.features)

    def entrenar(self, *args, **kwargs):
        """Entrena el modelo supervisado delegando según el tipo de tarea u objetivo."""
        self._reiniciar_resultados()
        algoritmo = kwargs.get("algoritmo")
        delegado = self._delegado_supervisado(algoritmo)
        if type(delegado) is type(self):
            raise ValueError("Especifique una subclase o algoritmo supervisado válido.")
        if algoritmo is None and delegado._tipo_target() == "clasificacion":
            kwargs = {"algoritmo": "RF", **kwargs}
        resultado = delegado.entrenar(*args, **kwargs)
        self.target = delegado.target
        self.features = delegado.features
        self.modelo = delegado.modelo
        self.resultado = delegado.resultado
        self._preparados = delegado._preparados
        return resultado

    def experimentar(self, *args, **kwargs):
        """Ejecuta un experimento comparativo supervisado sin usar test para decidir."""
        self._reiniciar_resultados()
        delegado = self._delegado_supervisado()
        if type(delegado) is type(self):
            raise ValueError("Especifique una subclase supervisada válida.")
        experimento = delegado.experimentar(*args, **kwargs)
        self.target = delegado.target
        self.features = delegado.features
        self.experimento = experimento
        self.resultado = delegado.resultado
        self.modelo = delegado.modelo
        self._preparados = delegado._preparados
        return experimento

    benchmark = experimentar

    def _transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma características con el preprocesador ajustado en train."""
        preparados = self._preparados
        if preparados is None and self.resultado is not None:
            preparados = getattr(self.resultado, "datos_preparados", None)
        if preparados is None:
            raise RuntimeError("No hay transformador disponible; entrene primero.")
        return preparados.transformador.transform(X)

    def evaluar(self):
        """Devuelve el resultado de evaluación del último modelo entrenado."""
        if self.resultado is None:
            raise RuntimeError(self._mensaje_sin_entrenar)
        return self.resultado

    def predecir(self, X=None):
        """Genera predicciones conservando el orden e índice de las observaciones."""
        if self.modelo is None and self.resultado is None:
            raise RuntimeError(self._mensaje_sin_entrenar)
        if X is None:
            return self.evaluar().y_pred.copy()
        if isinstance(X, pd.Series):
            raise TypeError("X debe ser una tabla de características (DataFrame).")
        features = (
            list(self.resultado.features)
            if self.resultado is not None and hasattr(self.resultado, "features")
            else list(self.features or ())
        )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features)
        faltantes = [c for c in features if c not in X]
        if faltantes:
            raise ValueError(f"Faltan variables predictoras: {faltantes}")
        modelo = self.modelo if self.modelo is not None else self.evaluar().modelo
        return pd.Series(
            modelo.predict(self._transformar(X)), index=X.index, name="prediccion"
        )

    def previsualizar_particion(
        self,
        *,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool | None = None,
        incluir_categoricas: bool = False,
        imputar: bool = True,
        estandarizar: bool = True,
        particion: ResultadoParticion | None = None,
    ) -> dict[str, Any]:
        """Describe exactamente los índices reutilizados después al entrenar."""
        base = self._datos_supervisados()
        if stratify is None:
            stratify = self._tipo_target() == "clasificacion"
        train, test, validacion = self._indices(
            base,
            particion=particion,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        seleccion = base[self.features]
        resumen = {
            "tam_train": len(train),
            "tam_test": len(test),
            "tam_validacion": len(validacion),
            "duplicados": int(seleccion.duplicated().sum()),
            "nulos": int(seleccion.isna().sum().sum()),
            "nulos_porcentaje": float(base.isna().mean().mean() * 100),
            "head": base.loc[train].head(5),
            "tail": base.loc[test].tail(5),
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "validacion_indices": validacion.tolist(),
        }
        if stratify or self._tipo_target() == "clasificacion":
            resumen["distribucion_train"] = (
                base.loc[train, self.target].value_counts().to_dict()
            )
            resumen["distribucion_test"] = (
                base.loc[test, self.target].value_counts().to_dict()
            )
        return resumen

    def __str__(self) -> str:
        self._definir_target()
        return (
            "Clase Supervisado\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
