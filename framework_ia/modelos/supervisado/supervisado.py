"""Clase base de aprendizaje supervisado con herencia y desacoplamiento de utilidades."""

from __future__ import annotations

import platform
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from ...datos.eda import EDA
from ...datos.particion import comprobar_clases, dividir_indices, huella_datos, validar_particion
from ...utils.resultados import ResultadoParticion
from ...utils.supervisado import (
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


class Supervisado(EDA):
    """Base común del aprendizaje supervisado.

    Proporciona validación estructural de ``target`` y ``features``, particiones
    sin fuga de datos, orquestación de entrenamiento y experimentación mediante
    herencia, preprocesamiento y contratos pythónicos.
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
        """Limpia artefactos previos."""
        self.resultado = None
        self.experimento = None
        self.modelo = None
        self._preparados = None
        self.exactitud = None
        self.r2 = None

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

    def _es_regresion(self) -> bool:
        """Comprueba si la instancia opera en modalidad regresión."""
        from .regresion import Regresion

        if isinstance(self, Regresion):
            return True
        from .clasificacion import Clasificacion

        if isinstance(self, Clasificacion):
            return False
        return self._tipo_target() == "regresion"

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
        """Obtiene los índices de train, test y validación sin fuga de datos."""
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
        """Valida que las predictoras sean aptas para el algoritmo."""
        if self._es_regresion():
            return validar_predictoras_regresion(self.features, base, algoritmo)
        return self.features

    def _stratify_defecto(self) -> bool:
        """Define si la estratificación se aplica por defecto en la tarea supervisada."""
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
        """Construye metadatos de trazabilidad comunes a todas las tareas supervisadas."""
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

    def _delegado_supervisado(self, algoritmo: str | None = None) -> "Supervisado":
        """Instancia la subclase correspondiente si se invoca Supervisado directamente."""
        from .clasificacion import Clasificacion
        from .regresion import Regresion

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

    def _normalizar_algoritmo(self, algoritmo: str | None) -> str:
        """Normaliza el identificador del algoritmo según la modalidad."""
        if self._es_regresion():
            alg = str(algoritmo or "RLM").strip()
            if alg.lower() in {
                "rl",
                "regresion_logistica",
                "logistic_regression",
                "logisticregression",
            }:
                from .clasificacion import Clasificacion

                return Clasificacion(
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
        """Valida que los conjuntos de partición cumplan los requisitos de la tarea."""
        if self._es_regresion():
            if len(test) < 2:
                raise ValueError(
                    "La evaluación necesita al menos dos filas para calcular R²."
                )
        else:
            comprobar_clases(base.loc[train, self.target], base.loc[test, self.target])

    def _ajustar(self, base, train, algoritmo, random_state, opciones, parametros):
        """Ajusta el estimador supervisado según la modalidad."""
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
        """Construye el contenedor tipado de resultados supervisados."""
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
        """Sincroniza los atributos escalares principales de evaluación."""
        if self._es_regresion():
            self.r2 = resultado.metricas.get("r2")
        else:
            self.exactitud = resultado.metricas.get("accuracy")

    def _ejecutar_experimento(self, **kwargs) -> Any:
        """Ejecuta la suite experimental adecuada a la tarea supervisada."""
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
        """Orquesta el flujo común de entrenamiento supervisado sin fuga de datos."""
        if type(self) is Supervisado:
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
        """Ejecuta un experimento comparativo supervisado sin fuga de datos."""
        if type(self) is Supervisado:
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
            raise RuntimeError("Entrene primero un modelo antes de evaluarlo.")
        return self.resultado

    def predecir(self, X=None):
        """Genera predicciones conservando el orden e índice de las observaciones."""
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
        """Describe exactamente los índices reutilizados después al entrenar."""
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

    def __str__(self) -> str:
        self._definir_target()
        return (
            "Clase Supervisado\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
