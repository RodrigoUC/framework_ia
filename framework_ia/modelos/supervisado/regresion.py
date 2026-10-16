"""Regresión supervisada con particiones reproducibles y evaluación numérica."""

from __future__ import annotations

import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ...resultados import ResultadoParticion, ResultadoRegresion
from ...utils.configuracion_regresion import ALGORITMOS_LAB3
from .algoritmos_regresion import (
    crear_modelo,
    metadatos_regresion,
    normalizar_algoritmo,
    parametros_regresion,
)
from .base import Supervisado
from .particiones import dividir_indices, huella_datos, validar_particion
from .preprocesamiento import PreprocesadorSupervisado


def metricas_regresion(y_true, y_pred):
    """Calcula errores en unidades originales y R² con al menos dos filas."""
    if len(y_true) < 2:
        raise ValueError("La evaluación necesita al menos dos filas para calcular R².")
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "r2": float(r2_score(y_true, y_pred)),
    }


class Regresion(Supervisado):
    """Modelos numéricos que aprenden transformaciones sólo de entrenamiento."""

    def __init__(
        self, dataframe=None, ruta_datos=None, *, target=None, features=None, **kwargs
    ):
        super().__init__(
            dataframe=dataframe,
            ruta_datos=ruta_datos,
            target=target,
            features=features,
            **kwargs,
        )
        self._reiniciar_resultados()

    def _reiniciar_resultados(self):
        self.resultado: ResultadoRegresion | None = None
        self.experimento = None
        self.modelo = None
        self._preparados = None
        self.r2 = None

    def _datos_regresion(self):
        if self.datos.empty:
            raise ValueError("No hay datos para entrenar.")
        if not self.datos.index.is_unique:
            raise ValueError("El dataset necesita índices únicos.")
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
        base = self.datos.loc[:, self.features + [self.target]].copy()
        self._validar_objetivo(base[self.target])
        return base

    @staticmethod
    def _validar_objetivo(y):
        if (
            not pd.api.types.is_numeric_dtype(y)
            or pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_complex_dtype(y)
        ):
            raise TypeError("La variable objetivo debe ser numérica real.")
        if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError(
                "La variable objetivo no admite valores nulos o no finitos."
            )
        if y.nunique() < 2:
            raise ValueError(
                "La variable objetivo debe tener variación; no ser constante."
            )

    def _indices(self, base, *, particion, test_size, random_state):
        if particion is not None:
            validar_particion(particion, self.datos)
            return tuple(
                subset.index.copy() if subset is not None else pd.Index([])
                for subset in (particion.train, particion.test, particion.validacion)
            )
        train, test = dividir_indices(
            base,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=False,
        )
        return train, test, pd.Index([])

    def _validar_predictoras(self, base, algoritmo):
        if algoritmo not in ALGORITMOS_LAB3:
            raise ValueError(f"Algoritmo de regresión desconocido: {algoritmo}")
        if algoritmo == "RLS":
            if len(self.features) != 1:
                raise ValueError(
                    "RLS requiere exactamente una variable predictora numérica."
                )
            serie = base[self.features[0]]
            if (
                not pd.api.types.is_numeric_dtype(serie)
                or pd.api.types.is_bool_dtype(serie)
                or pd.api.types.is_complex_dtype(serie)
            ):
                raise TypeError("RLS requiere una variable predictora numérica real.")
        elif algoritmo == "RLM" and len(self.features) < 2:
            raise ValueError("RLM requiere al menos dos variables predictoras.")

    def _ajustar(self, base, train, algoritmo, random_state, opciones, parametros):
        self._validar_predictoras(base, algoritmo)
        if len(train) < 2:
            raise ValueError("Entrenamiento necesita al menos dos filas.")
        y_train = base.loc[train, self.target]
        self._validar_objetivo(y_train)
        preprocesador = PreprocesadorSupervisado(
            **{**opciones, "random_state": random_state, "seleccion_k": None}
        )
        preparados = preprocesador.ajustar_transformar(
            base.loc[train, self.features], y_train
        )
        modelo = crear_modelo(algoritmo, random_state, parametros)
        modelo.fit(preparados.matriz, y_train)
        return modelo, preparados

    def _resultado(
        self, base, train, evaluacion, algoritmo, modelo, preparados, metadatos
    ):
        def predecir(indices):
            X = preparados.transformador.transform(base.loc[indices, self.features])
            return pd.Series(modelo.predict(X), index=indices, name="prediccion")

        y_train = base.loc[train, self.target].rename("real").copy()
        y_true = base.loc[evaluacion, self.target].rename("real").copy()
        y_pred_train, y_pred = predecir(train), predecir(evaluacion)
        return ResultadoRegresion(
            algoritmo=algoritmo,
            target=self.target,
            features=tuple(self.features),
            y_true=y_true,
            y_pred=y_pred,
            y_train=y_train,
            y_pred_train=y_pred_train,
            residuos=(y_true - y_pred).rename("residuo"),
            residuos_train=(y_train - y_pred_train).rename("residuo"),
            metricas=metricas_regresion(y_true, y_pred),
            metricas_train=metricas_regresion(y_train, y_pred_train),
            muestra_train=len(train),
            muestra_test=len(evaluacion),
            modelo=modelo,
            datos_preparados=preparados,
            parametros=parametros_regresion(modelo),
            metadatos={
                **metadatos,
                **metadatos_regresion(modelo),
                "columnas_transformadas": list(preparados.matriz.columns),
                "columnas_descartadas": list(preparados.columnas_descartadas),
            },
        )

    def _metadatos(self, train, test, validacion, *, random_state, particion, opciones):
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
            "conjunto_evaluacion": "test",
            "preprocesamiento": {
                **opciones,
                "seleccion_k": None,
                "ajustado_solo_train": True,
            },
            "versiones": {
                "python": platform.python_version(),
                **{
                    nombre: version(nombre)
                    for nombre in ("numpy", "pandas", "scikit-learn")
                },
            },
        }

    def entrenar(
        self,
        *,
        algoritmo="RLM",
        test_size=0.25,
        random_state=42,
        incluir_categoricas=False,
        imputar=True,
        estandarizar=True,
        particion: ResultadoParticion | None = None,
        **kwargs,
    ):
        """Separa filas antes de ajustar; validación externa queda reservada."""
        algoritmo = normalizar_algoritmo(algoritmo)
        base = self._datos_regresion()
        train, test, validacion = self._indices(
            base, particion=particion, test_size=test_size, random_state=random_state
        )
        if len(test) < 2:
            raise ValueError(
                "La evaluación necesita al menos dos filas para calcular R²."
            )
        opciones = {
            "incluir_categoricas": incluir_categoricas,
            "imputar": imputar,
            "estandarizar": estandarizar,
        }
        modelo, preparados = self._ajustar(
            base, train, algoritmo, random_state, opciones, kwargs
        )
        metadatos = self._metadatos(
            train,
            test,
            validacion,
            random_state=random_state,
            particion=particion,
            opciones=opciones,
        )
        resultado = self._resultado(
            base, train, test, algoritmo, modelo, preparados, metadatos
        )
        self.resultado, self.modelo, self._preparados = resultado, modelo, preparados
        self.r2 = resultado.metricas["r2"]
        return resultado

    def predecir(self, X=None):
        """Predice sin reajustar y conserva el orden e índice de las filas."""
        resultado = self.evaluar()
        if X is None:
            return resultado.y_pred.copy()
        if isinstance(X, pd.Series):
            raise TypeError("X debe ser una tabla de características (DataFrame).")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=resultado.features)
        faltantes = [c for c in resultado.features if c not in X]
        if faltantes:
            raise ValueError(f"Faltan variables predictoras: {faltantes}")
        transformados = resultado.datos_preparados.transformador.transform(X)
        return pd.Series(
            resultado.modelo.predict(transformados), index=X.index, name="prediccion"
        )

    def experimentar(
        self,
        *,
        algoritmos=None,
        configuraciones=None,
        feature_simple=None,
        particion=None,
        test_size=0.25,
        validation_size=0.2,
        random_state=42,
        estrategia=None,
        folds=5,
        incluir_categoricas=False,
        imputar=True,
        estandarizar=True,
    ):
        """Selecciona por RMSE de validación sin usar prueba para decidir.

        Una partición con validación externa usa holdout por defecto; en otro
        caso se aplica CV sólo sobre train. Un intento sin ganadores devuelve
        su evidencia sin sustituir el último experimento exitoso.
        """
        from .experimentos_regresion import ejecutar_experimento

        intento = ejecutar_experimento(
            self,
            algoritmos=algoritmos,
            configuraciones=configuraciones,
            feature_simple=feature_simple,
            particion=particion,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            estrategia=estrategia,
            folds=folds,
            incluir_categoricas=incluir_categoricas,
            imputar=imputar,
            estandarizar=estandarizar,
        )
        if intento.mejores:
            self.experimento = intento
            elegido = intento.mejores.get(intento.mejor_algoritmo)
            if elegido is not None:
                self.resultado = elegido
                self.modelo = elegido.modelo
                self._preparados = elegido.datos_preparados
                self.r2 = elegido.metricas["r2"]
        return intento

    benchmark = experimentar

    def evaluar(self):
        """Devuelve la última evaluación sin volver a entrenar."""
        if self.resultado is None:
            raise RuntimeError("Entrene primero un regresor.")
        return self.resultado

    def lineal_simple(self, **kwargs):
        return self.entrenar(algoritmo="RLS", **kwargs)

    def lineal_multiple(self, **kwargs):
        return self.entrenar(algoritmo="RLM", **kwargs)

    RLS = lineal_simple
    RLM = lineal_multiple

    def lasso(self, **kwargs):
        return self.entrenar(algoritmo="Lasso", **kwargs)

    def lasso_cv(self, **kwargs):
        return self.entrenar(algoritmo="LassoCV", **kwargs)

    def ridge(self, **kwargs):
        return self.entrenar(algoritmo="Ridge", **kwargs)

    def ridge_cv(self, **kwargs):
        return self.entrenar(algoritmo="RidgeCV", **kwargs)

    def svr(self, **kwargs):
        return self.entrenar(algoritmo="SVR", **kwargs)

    def arbol_decision(self, **kwargs):
        return self.entrenar(algoritmo="DT", **kwargs)

    def random_forest(self, **kwargs):
        return self.entrenar(algoritmo="RF", **kwargs)

    Lasso = lasso
    LassoCV = lasso_cv
    Ridge = ridge
    RidgeCV = ridge_cv
    SVR = svr
    DT = arbol_decision
    RF = random_forest

    def RL(self, *args, **kwargs):
        """Regresión Logística.

        TODO: implementar la regresión logística (modelo de clasificación).
        """
        raise NotImplementedError("Debe implementar el método RL.")

    def __str__(self) -> str:
        self._definir_target()
        base = (
            "Clase Regresion\n"
            f"Target: {self.target} ({self._tipo_target()})\n"
            f"Features: {self._definir_features()}"
        )
        if self.r2 is not None:
            base += f"\nR²: {self.r2:.4f}"
        return base


Progresion = Regresion
