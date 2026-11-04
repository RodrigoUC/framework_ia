"""Base supervised learning class with inheritance and decoupled utilities."""

from __future__ import annotations

import platform
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from ai_framework.data.eda import EDA
from ai_framework.data.partition import comprobar_clases, dividir_indices, huella_datos, validar_particion
from ai_framework.utils.results import ResultadoParticion
from ai_framework.utils.supervised import (
    ajustar_clasificacion,
    ajustar_regresion,
    crear_resultado_clasificacion,
    crear_resultado_regresion,
    ejecutar_experimento_clasificacion,
    ejecutar_experimento_regresion,
    metricas_clasificacion,
    normalizar_algoritmo_clasificacion,
    normalizar_algoritmo_regresion,
    ordenar_labels,
    validar_datos_clasificacion,
    validar_datos_regresion,
    validar_objetivo_regresion,
    validar_predictoras_regresion,
)


class Supervised(EDA):
    """Common base for supervised learning.

    Provides structural validation of ``target`` and ``features``, leak-free
    partitions, orchestration of training and experimentation through inheritance,
    preprocessing, and pythonic contracts.
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
        self.target = target
        self.features = features
        self.modelo = None
        self.resultado: Any = None
        self.experimento: Any = None
        self._preparados: Any = None
        self.exactitud: float | None = None
        self.r2: float | None = None

    def _reiniciar_resultados(self) -> None:
        """Clean up previous artifacts."""
        self.resultado = None
        self.experimento = None
        self.modelo = None
        self._preparados = None
        self.exactitud = None
        self.r2 = None

    def _definir_target(self):
        """Return the target column name, defaulting to the last column if omitted."""
        if self.target is None:
            self.target = self.datos.columns[-1]
        return self.target

    def _tipo_target(self):
        """Return classification if target is categorical or regression if numerical."""
        self._definir_target()
        serie = self.datos[self.target]
        if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(
            serie
        ):
            return "regresion"
        return "clasificacion"

    def _es_regresion(self) -> bool:
        """Check if instance operates in regression mode."""
        if self.__class__.__name__ in ("Regresion", "Regression"):
            return True
        if self.__class__.__name__ in ("Clasificacion", "Classification"):
            return False
        from .regression import Regression

        if isinstance(self, Regression):
            return True
        from .classification import Classification

        if isinstance(self, Classification):
            return False
        return self._tipo_target() == "regresion"

    def _definir_features(self):
        """Return features (all columns except target)."""
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
        """Validate tabular integrity common to classification and regression."""
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
        """Obtain leak-free train, test, and validation indices."""
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
        """Return the validated base DataFrame according to the supervised task."""
        if self._es_regresion():
            base = self._validar_estructura_base(
                mensaje_indices="El dataset necesita índices únicos."
            )
            return validar_datos_regresion(base, self.target)
        base = self._validar_estructura_base(
            mensaje_indices="El dataset necesita índices únicos; restablezca su índice antes de partir."
        )
        return validar_datos_clasificacion(base, self.target)

    _datos_regresion = _datos_supervisados
    _datos_clasificacion = _datos_supervisados

    _validar_objetivo = staticmethod(validar_objetivo_regresion)
    _ordenar_labels = staticmethod(ordenar_labels)
    _metricas_supervisadas = staticmethod(metricas_clasificacion)

    def _validar_predictoras(self, base: pd.DataFrame, algoritmo: str | None = None):
        """Validate predictor features for the specified algorithm."""
        if self._es_regresion():
            return validar_predictoras_regresion(self.features, base, algoritmo)
        return self.features

    def _stratify_defecto(self) -> bool:
        """Determine whether stratification applies by default for the supervised task."""
        return not self._es_regresion()

    def _metadatos(
        self,
        train: pd.Index,
        test: pd.Index,
        validacion: pd.Index,
        *,
        random_state: int,
        particion: ResultadoParticion | None,
        opciones: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct traceability metadata common to all supervised tasks."""
        versiones = {
            "python": platform.python_version(),
            **{
                nombre: version(nombre)
                for nombre in ("numpy", "pandas", "scikit-learn")
            },
        }
        return {
            "semilla": random_state,
            "semilla_particion": particion.semilla
            if particion is not None
            else random_state,
            "particion_externa": particion is not None,
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "validacion_indices": validacion.tolist(),
            "huella_datos": huella_datos(self.datos),
            "target": self.target,
            "features": list(self.features),
            "filas_objetivo_nulo": int(
                self.datos[self.target].replace([np.inf, -np.inf], np.nan).isna().sum()
            ),
            "conjunto_evaluacion": "test",
            "preprocesamiento": {
                "seleccion_k": None,
                **opciones,
                "ajustado_solo_train": True,
            },
            "versiones": versiones,
        }

    def _delegado_supervisado(self, algoritmo: str | None = None) -> "Supervised":
        """Instantiate the corresponding subclass if Supervised is invoked directly."""
        from .classification import Classification
        from .regression import Regression

        if algoritmo is not None:
            alg_norm = str(algoritmo).strip().lower()
            if alg_norm in {
                "knn",
                "xgb",
                "xgboost",
                "adaboost",
                "ada",
                "nr",
                "nb",
                "naive_bayes",
                "rl",
                "regresion_logistica",
                "logistic_regression",
            }:
                return Classification(
                    dataframe=self.datos, target=self.target, features=self.features
                )
            if alg_norm in {
                "rls",
                "rlm",
                "lineal_simple",
                "lineal_multiple",
                "simple_linear_regression",
                "multiple_linear_regression",
                "lasso",
                "lassocv",
                "lasso_cv",
                "ridge",
                "ridgecv",
                "ridge_cv",
                "svr",
            }:
                return Regression(
                    dataframe=self.datos, target=self.target, features=self.features
                )
        clase = Regression if self._tipo_target() == "regresion" else Classification
        return clase(dataframe=self.datos, target=self.target, features=self.features)

    def _normalizar_algoritmo(self, algoritmo: str | None) -> str:
        """Normalize the algorithm identifier based on task mode."""
        if self._es_regresion():
            alg = str(algoritmo or "RLM").strip()
            if alg.lower() in {
                "rl",
                "regresion_logistica",
                "logistic_regression",
                "logisticregression",
            }:
                from .classification import Classification

                return Classification(
                    dataframe=self.datos, target=self.target, features=self.features
                ).RL
            return normalizar_algoritmo_regresion(alg)
        return normalizar_algoritmo_clasificacion(algoritmo or "RF")

    def _validar_entrenamiento(
        self,
        base: pd.DataFrame,
        train: pd.Index,
        test: pd.Index,
        validacion: pd.Index,
    ) -> None:
        """Validate partition sets against task requirements."""
        if self._es_regresion():
            if len(test) < 2:
                raise ValueError(
                    "La evaluación necesita al menos dos filas para calcular R²."
                )
        else:
            comprobar_clases(base.loc[train, self.target], base.loc[test, self.target])

    def _ajustar(self, base, train, algoritmo, random_state, opciones, parametros):
        """Fit the supervised estimator according to the mode."""
        if self._es_regresion():
            return ajustar_regresion(
                base,
                train,
                self.features,
                self.target,
                algoritmo,
                random_state,
                opciones,
                parametros,
            )
        return ajustar_clasificacion(
            base,
            train,
            self.features,
            self.target,
            algoritmo,
            random_state,
            opciones,
            parametros,
        )

    def _resultado(
        self, base, train, evaluacion, algoritmo, modelo, preparados, metadatos
    ):
        """Build the typed supervised result container."""
        if self._es_regresion():
            return crear_resultado_regresion(
                base,
                train,
                evaluacion,
                self.features,
                self.target,
                algoritmo,
                modelo,
                preparados,
                metadatos,
            )
        return crear_resultado_clasificacion(
            base,
            train,
            evaluacion,
            self.features,
            self.target,
            algoritmo,
            modelo,
            preparados,
            metadatos,
        )

    def _actualizar_metricas_resumen(self, resultado: Any) -> None:
        """Synchronize main scalar evaluation metrics."""
        if self._es_regresion():
            self.r2 = resultado.metricas.get("r2")
        else:
            self.exactitud = resultado.metricas.get("accuracy")

    def _ejecutar_experimento(self, **kwargs) -> Any:
        """Execute the appropriate experimental suite for the supervised task."""
        if self._es_regresion():
            intento = ejecutar_experimento_regresion(self, **kwargs)
            elegido = intento.mejores.get(intento.mejor_algoritmo)
            if elegido is not None:
                self.experimento = intento
                self.resultado = elegido
                self.modelo = elegido.modelo
                self._preparados = elegido.datos_preparados
                self.r2 = elegido.metricas["r2"]
            return intento
        self._reiniciar_resultados()
        intento = ejecutar_experimento_clasificacion(self, **kwargs)
        self.experimento = intento
        return intento

    def entrenar(
        self,
        *,
        algoritmo: str | None = None,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool | None = None,
        incluir_categoricas: bool = False,
        imputar: bool = True,
        estandarizar: bool = True,
        particion: ResultadoParticion | None = None,
        **kwargs,
    ):
        """Orchestrate common leak-free supervised training workflow."""
        if type(self) is Supervised:
            delegado = self._delegado_supervisado(algoritmo)
            if type(delegado) is type(self):
                raise ValueError("Especifique una subclase o algoritmo supervisado válido.")
            if algoritmo is None and delegado._tipo_target() == "clasificacion":
                algoritmo = "RF"
            resultado = delegado.entrenar(
                algoritmo=algoritmo,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
                incluir_categoricas=incluir_categoricas,
                imputar=imputar,
                estandarizar=estandarizar,
                particion=particion,
                **kwargs,
            )
            self.target = delegado.target
            self.features = delegado.features
            self.modelo = delegado.modelo
            self.resultado = delegado.resultado
            self._preparados = delegado._preparados
            self.exactitud = delegado.exactitud
            self.r2 = delegado.r2
            return resultado

        alg_norm = self._normalizar_algoritmo(algoritmo)
        base = self._datos_supervisados()
        if stratify is None:
            stratify = self._stratify_defecto()
        train, test, validacion = self._indices(
            base,
            particion=particion,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        self._validar_entrenamiento(base, train, test, validacion)
        opciones = {
            "incluir_categoricas": incluir_categoricas,
            "imputar": imputar,
            "estandarizar": estandarizar,
        }
        modelo, preparados = self._ajustar(
            base, train, alg_norm, random_state, opciones, kwargs
        )
        metadatos = self._metadatos(
            train,
            test,
            validacion,
            random_state=random_state,
            particion=particion,
            opciones=opciones,
        )
        if alg_norm == "XGBoost":
            try:
                metadatos["versiones"]["xgboost"] = version("xgboost")
            except Exception:
                pass
        resultado = self._resultado(
            base, train, test, alg_norm, modelo, preparados, metadatos
        )
        self.resultado = resultado
        self.modelo = modelo
        self._preparados = preparados
        self._actualizar_metricas_resumen(resultado)
        return resultado

    def experimentar(self, *args, **kwargs):
        """Execute a leak-free comparative supervised experiment."""
        if type(self) is Supervised:
            delegado = self._delegado_supervisado()
            if type(delegado) is type(self):
                raise ValueError("Especifique una subclase supervisada válida.")
            experimento = delegado.experimentar(*args, **kwargs)
            self.target = delegado.target
            self.features = delegado.features
            self.experimento = delegado.experimento
            self.resultado = delegado.resultado
            self.modelo = delegado.modelo
            self._preparados = delegado._preparados
            self.exactitud = delegado.exactitud
            self.r2 = delegado.r2
            return experimento

        return self._ejecutar_experimento(*args, **kwargs)

    benchmark = experimentar
    experiment = experimentar
    train = entrenar

    def _transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the preprocessor fitted only on train."""
        preparados = self._preparados
        if preparados is None and self.resultado is not None:
            preparados = getattr(self.resultado, "datos_preparados", None)
        if preparados is None:
            raise RuntimeError("No hay transformador disponible; entrene primero.")
        return preparados.transformador.transform(X)

    def evaluar(self):
        """Return evaluation result of the latest trained model."""
        if self.resultado is None:
            raise RuntimeError("Entrene primero un modelo antes de evaluarlo.")
        return self.resultado

    def predecir(self, X=None):
        """Generate predictions preserving observation order and index."""
        if self.modelo is None and self.resultado is None:
            raise RuntimeError("Entrene primero un modelo antes de predecir.")
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
        """Describe exact indices reused later during training."""
        base = self._datos_supervisados()
        if stratify is None:
            stratify = self._stratify_defecto()
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

    evaluate = evaluar
    predict = predecir
    preview_partition = previsualizar_particion

    def __str__(self) -> str:
        self._definir_target()
        return (
            "Class Supervised\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )


def __getattr__(name: str):
    if name == "Supervisado":
        import ai_framework.utils.aliases as aliases
        return aliases.Supervisado
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
