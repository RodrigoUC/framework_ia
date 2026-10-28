"""Mecanismo unificado de experimentación y selección de modelos supervisados."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ai_framework.data.partition import comprobar_clases, dividir_indices
from ..results import (
    ResultadoExperimentoClasificacion,
    ResultadoExperimentoRegresion,
)
from .classification_config import (
    METRICAS_SELECCION as METRICAS_SELECCION_CLASIFICACION,
)
from .regression_config import (
    ALGORITMOS_LAB3 as ALGORITMOS_REGRESION_DEFECTO,
    configuraciones_experimento_regresion,
)
from .regression_algorithms import (
    metadatos_regresion,
    normalizar_algoritmo as normalizar_regresion,
    parametros_regresion,
)
from .classification import (
    ALGORITMOS_LAB2 as ALGORITMOS_CLASIFICACION_DEFECTO,
    configuraciones_lab2,
    normalizar_algoritmo as normalizar_clasificacion,
)
from .regression import metricas_regresion

METRICAS_SELECCION = tuple(METRICAS_SELECCION_CLASIFICACION)

COLUMNAS_TABLA_CLASIFICACION = (
    "algoritmo",
    "configuracion",
    "parametros",
    "accuracy_validacion",
    "precision_macro_validacion",
    "recall_macro_validacion",
    "f1_macro_validacion",
    "puntuacion_validacion",
    "seleccionado",
    "estado",
    "error",
)

COLUMNAS_TABLA_REGRESION = (
    "algoritmo",
    "configuracion",
    "features",
    "parametros",
    "rmse_validacion",
    "mae_validacion",
    "mse_validacion",
    "r2_validacion",
    "puntuacion_validacion",
    "puntuacion_validacion_std",
    "puntuaciones_pliegues",
    "metricas_pliegues",
    "seleccionado",
    "estado",
    "error",
)


def _parametros_registrados(efectivos: dict, solicitados: dict) -> dict:
    """Parámetros del estimador más las opciones propias del framework."""
    registrados = dict(efectivos)
    if solicitados.get("balancear_clases"):
        registrados["balancear_clases"] = True
    return registrados


def _candidatos_clasificacion(algoritmos, configuraciones):
    exp = ExperimentoClasificacion()
    return exp._normalizar_catalogo(
        algoritmos=algoritmos,
        configuraciones=configuraciones,
        catalogo_base=exp._catalogo_base_clasificacion(),
        normalizador_fn=normalizar_clasificacion,
        algoritmos_defecto=ALGORITMOS_CLASIFICACION_DEFECTO,
    )


def _evaluar(regresor, base, pliegues, algoritmo, random_state, opciones, parametros):
    """Evalúa pliegues de validación para regresión."""
    metricas = []
    modelo = preparados = None
    for numero, (train, validacion) in enumerate(pliegues, start=1):
        modelo, preparados = regresor._ajustar(
            base, train, algoritmo, random_state, opciones, parametros
        )
        X = preparados.transformador.transform(base.loc[validacion, regresor.features])
        valores = metricas_regresion(
            base.loc[validacion, regresor.target], modelo.predict(X)
        )
        if not all(np.isfinite(valor) for valor in valores.values()):
            raise ValueError("La evaluación produjo métricas no finitas.")
        metricas.append({"pliegue": numero, **valores, **metadatos_regresion(modelo)})
    return metricas, modelo, preparados


def _particiones(
    regresor,
    base,
    *,
    particion,
    test_size,
    validation_size,
    random_state,
    estrategia,
    folds,
):
    """Genera pliegues y valida estrategia de partición para regresión."""
    train, test, validacion = regresor._indices(
        base, particion=particion, test_size=test_size, random_state=random_state
    )
    estrategia = (
        ("holdout" if not validacion.empty else "cv")
        if estrategia is None
        else estrategia
    )
    if estrategia not in ("cv", "holdout"):
        raise ValueError("estrategia debe ser 'cv', 'holdout' o None (automática).")
    if len(test) < 2:
        raise ValueError("Prueba necesita al menos dos filas para calcular R².")
    if estrategia == "holdout":
        if validacion.empty:
            train, validacion = dividir_indices(
                base.loc[train],
                regresor.target,
                test_size=validation_size,
                random_state=random_state,
                stratify=False,
            )
        pliegues = [(train, validacion)]
    else:
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
            raise ValueError("folds debe ser un entero mayor o igual que dos.")
        if len(train) < 2 * folds:
            raise ValueError(
                "Cada pliegue de validación necesita al menos dos filas para R²."
            )
        divisor = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        pliegues = [(train.take(t), train.take(v)) for t, v in divisor.split(train)]
    for entrenamiento, evaluacion in pliegues:
        if len(entrenamiento) < 2 or len(evaluacion) < 2:
            raise ValueError("Entrenamiento y validación necesitan al menos dos filas.")
        regresor._validar_objetivo(base.loc[entrenamiento, regresor.target])
    return train, test, validacion, estrategia, pliegues


class ExperimentoSupervisado:
    """Base común para la ejecución de benchmarks y comparación de modelos."""

    def _normalizar_catalogo(
        self,
        *,
        algoritmos: list[str] | str | None,
        configuraciones: dict[str, list[dict[str, Any]]] | None,
        catalogo_base: dict[str, list[dict[str, Any]]],
        normalizador_fn: Any,
        algoritmos_defecto: tuple[str, ...],
    ) -> dict[str, list[dict[str, Any]]]:
        if isinstance(algoritmos, str):
            algoritmos = [algoritmos]
        elegidos = list(
            dict.fromkeys(
                normalizador_fn(a)
                for a in (algoritmos_defecto if algoritmos is None else algoritmos)
            )
        )
        if not elegidos:
            raise ValueError("Seleccione al menos un algoritmo para experimentar.")
        catalogo = dict(catalogo_base)
        if configuraciones is not None:
            if not isinstance(configuraciones, dict):
                raise TypeError("configuraciones debe ser un diccionario por algoritmo.")
            vistos = set()
            for clave, lista in configuraciones.items():
                algoritmo = normalizador_fn(clave)
                if algoritmo in vistos:
                    raise ValueError(
                        f"Configuraciones repetidas para {algoritmo} mediante aliases."
                    )
                vistos.add(algoritmo)
                catalogo[algoritmo] = lista
        resultado = {}
        for algoritmo in elegidos:
            lista = catalogo.get(algoritmo)
            if not isinstance(lista, (list, tuple)) or not lista:
                raise ValueError(f"{algoritmo} necesita al menos una configuración.")
            nombres = set()
            resultado[algoritmo] = []
            for config in lista:
                if not isinstance(config, dict) or set(config) != {"nombre", "parametros"}:
                    raise ValueError(
                        "Cada configuración necesita exactamente nombre y parametros."
                    )
                nombre, parametros = config["nombre"], config["parametros"]
                if not isinstance(nombre, str) or not nombre.strip() or nombre in nombres:
                    raise ValueError(
                        f"Nombres de configuración vacíos o repetidos para {algoritmo}."
                    )
                if not isinstance(parametros, dict) or not all(
                    isinstance(c, str) for c in parametros
                ):
                    raise ValueError(
                        "parametros debe ser un diccionario de argumentos del estimador."
                    )
                if {"eval_set", "callbacks", "early_stopping_rounds"} & set(parametros):
                    raise ValueError(
                        "El benchmark no admite eval_set, callbacks ni early_stopping_rounds externos."
                    )
                nombres.add(nombre)
                resultado[algoritmo].append(
                    {"nombre": nombre, "parametros": dict(parametros)}
                )
        return resultado


class ExperimentoClasificacion(ExperimentoSupervisado):
    """Ejecución de experimentos comparativos para clasificación."""

    def _catalogo_base_clasificacion(self) -> dict[str, list[dict[str, Any]]]:
        catalogo = configuraciones_lab2()
        catalogo["NR"] = [{"nombre": "estandar", "parametros": {}}]
        catalogo["RL"] = [{"nombre": "estandar", "parametros": {}}]
        return catalogo

    def ejecutar(
        self,
        clasificador,
        *,
        algoritmos=None,
        configuraciones=None,
        particion=None,
        test_size=0.25,
        validation_size=0.2,
        random_state=42,
        stratify=True,
        metrica="f1_macro",
        incluir_categoricas=False,
        imputar=True,
        estandarizar=True,
        estrategia="holdout",
        folds=5,
        seleccion_k="auto",
    ) -> ResultadoExperimentoClasificacion:
        if metrica not in METRICAS_SELECCION:
            raise ValueError(
                f"Métrica de selección no soportada: {metrica}. Use {METRICAS_SELECCION}."
            )
        if estrategia == "cv":
            from .cross_validation import ejecutar_experimento_cv

            return ejecutar_experimento_cv(
                clasificador,
                algoritmos=algoritmos,
                configuraciones=configuraciones,
                particion=particion,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
                metrica=metrica,
                incluir_categoricas=incluir_categoricas,
                imputar=imputar,
                estandarizar=estandarizar,
                folds=folds,
                seleccion_k=seleccion_k,
            )
        if estrategia != "holdout":
            raise ValueError("estrategia debe ser 'holdout' o 'cv'.")
        catalogo = self._normalizar_catalogo(
            algoritmos=algoritmos,
            configuraciones=configuraciones,
            catalogo_base=self._catalogo_base_clasificacion(),
            normalizador_fn=normalizar_clasificacion,
            algoritmos_defecto=ALGORITMOS_CLASIFICACION_DEFECTO,
        )
        base = clasificador._datos_clasificacion()
        train, test, validacion = clasificador._indices(
            base,
            particion=particion,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        validacion_externa = not validacion.empty
        if not validacion_externa:
            if not 0 < validation_size < 1:
                raise ValueError("validation_size debe estar entre 0 y 1 (excluido).")
            train, validacion = dividir_indices(
                base.loc[train],
                clasificador.target,
                test_size=validation_size,
                random_state=random_state,
                stratify=stratify,
            )
        comprobar_clases(
            base.loc[train, clasificador.target],
            base.loc[validacion, clasificador.target],
            base.loc[test, clasificador.target],
        )
        opciones = {
            "incluir_categoricas": incluir_categoricas,
            "imputar": imputar,
            "estandarizar": estandarizar,
        }
        metadatos = clasificador._metadatos(
            train,
            test,
            validacion,
            random_state=random_state,
            particion=particion,
            opciones=opciones,
        )
        metadatos.update(
            {
                "estrategia_seleccion": "holdout",
                "metrica_seleccion": metrica,
                "validacion_externa": validacion_externa,
                "validation_size_sobre_train": None
                if validacion_externa
                else validation_size,
                "reajustado_train_validacion": False,
                "test_usado_para_seleccion": False,
                "desempate": "primera configuracion en el orden declarado; luego primer algoritmo",
                "algoritmos": list(catalogo),
                "configuraciones": catalogo,
                "muestra_train": len(train),
                "muestra_validacion": len(validacion),
                "muestra_test": len(test),
                "distribuciones": {
                    nombre: base.loc[indices, clasificador.target].value_counts().to_dict()
                    for nombre, indices in (
                        ("train", train),
                        ("validacion", validacion),
                        ("test", test),
                    )
                },
            }
        )
        if "XGBoost" in catalogo:
            try:
                metadatos["versiones"]["xgboost"] = version("xgboost")
            except PackageNotFoundError:
                metadatos["versiones"]["xgboost"] = "no instalado"
        filas = []
        ganadores = {}
        for algoritmo, configs in catalogo.items():
            for config in configs:
                fila: dict[str, Any] = {c: None for c in COLUMNAS_TABLA_CLASIFICACION}
                fila.update(
                    algoritmo=algoritmo,
                    configuracion=config["nombre"],
                    parametros=dict(config["parametros"]),
                    seleccionado=False,
                    estado="ok",
                    error=None,
                )
                try:
                    modelo, preparados = clasificador._ajustar(
                        base,
                        train,
                        algoritmo,
                        random_state,
                        opciones,
                        config["parametros"],
                    )
                    validado = clasificador._resultado(
                        base,
                        train,
                        validacion,
                        algoritmo,
                        modelo,
                        preparados,
                        {"conjunto_evaluacion": "validacion"},
                    )
                    fila["parametros"] = _parametros_registrados(
                        validado.parametros, config["parametros"]
                    )
                    scores = {
                        "accuracy": validado.metricas["accuracy"],
                        **{
                            f"{nombre}_macro": validado.metricas[nombre]["macro"]
                            for nombre in ("precision", "recall", "f1")
                        },
                    }
                    fila.update(
                        {f"{nombre}_validacion": valor for nombre, valor in scores.items()}
                    )
                    fila["puntuacion_validacion"] = scores[metrica]
                    anterior = ganadores.get(algoritmo)
                    if anterior is None or scores[metrica] > anterior["score"]:
                        ganadores[algoritmo] = {
                            "score": scores[metrica],
                            "config": config,
                            "fila": len(filas),
                            "modelo": modelo,
                            "preparados": preparados,
                            "metricas_validacion": validado.metricas,
                        }
                except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                    fila.update(estado="error", error=f"{type(exc).__name__}: {exc}")
                filas.append(fila)

        mejor_algoritmo = max(ganadores, key=lambda a: ganadores[a]["score"], default=None)
        metadatos["selecciones"] = {a: g["config"]["nombre"] for a, g in ganadores.items()}
        metadatos["mejor_algoritmo_por_validacion"] = mejor_algoritmo
        metadatos["errores_prueba"] = {}
        tabla = pd.DataFrame(filas, columns=COLUMNAS_TABLA_CLASIFICACION)
        for ganador in ganadores.values():
            tabla.loc[ganador["fila"], "seleccionado"] = True
        metadatos["candidatos_exitosos"] = int((tabla["estado"] == "ok").sum())
        metadatos["candidatos_fallidos"] = int((tabla["estado"] == "error").sum())
        mejores = {}
        for algoritmo, ganador in ganadores.items():
            try:
                resultado = clasificador._resultado(
                    base,
                    train,
                    test,
                    algoritmo,
                    ganador["modelo"],
                    ganador["preparados"],
                    {
                        **metadatos,
                        "conjunto_evaluacion": "test",
                        "configuracion": ganador["config"]["nombre"],
                        "puntuacion_validacion": ganador["score"],
                        "metricas_validacion": ganador["metricas_validacion"],
                    },
                )
                mejores[algoritmo] = resultado
            except (ValueError, TypeError, RuntimeError) as exc:
                metadatos["errores_prueba"][algoritmo] = f"{type(exc).__name__}: {exc}"
        if mejor_algoritmo in mejores:
            clasificador.resultado = mejores[mejor_algoritmo]
            clasificador.modelo = ganadores[mejor_algoritmo]["modelo"]
            clasificador._preparados = ganadores[mejor_algoritmo]["preparados"]
            clasificador.exactitud = clasificador.resultado.metricas["accuracy"]
        return ResultadoExperimentoClasificacion(
            tabla=tabla,
            mejores=mejores,
            mejor_algoritmo=mejor_algoritmo,
            metadatos=metadatos,
        )


class ExperimentoRegresion(ExperimentoSupervisado):
    """Ejecución de experimentos comparativos para regresión."""

    def _predictora_simple(self, regresor, base, feature_simple):
        numericas = [
            columna
            for columna in regresor.features
            if pd.api.types.is_numeric_dtype(base[columna])
            and not pd.api.types.is_bool_dtype(base[columna])
            and not pd.api.types.is_complex_dtype(base[columna])
        ]
        if feature_simple is not None and feature_simple not in numericas:
            raise ValueError(
                "feature_simple debe ser una predictora numérica real seleccionada."
            )
        return feature_simple if feature_simple is not None else next(iter(numericas), None)

    def ejecutar(
        self,
        regresor,
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
    ) -> ResultadoExperimentoRegresion:
        base = regresor._datos_regresion()
        catalogo = self._normalizar_catalogo(
            algoritmos=algoritmos,
            configuraciones=configuraciones,
            catalogo_base=configuraciones_experimento_regresion(),
            normalizador_fn=normalizar_regresion,
            algoritmos_defecto=ALGORITMOS_REGRESION_DEFECTO,
        )
        simple = self._predictora_simple(regresor, base, feature_simple)
        train, test, validacion, estrategia, pliegues = _particiones(
            regresor,
            base,
            particion=particion,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            estrategia=estrategia,
            folds=folds,
        )
        opciones = {
            "incluir_categoricas": incluir_categoricas,
            "imputar": imputar,
            "estandarizar": estandarizar,
        }
        metadatos = regresor._metadatos(
            train,
            test,
            validacion,
            random_state=random_state,
            particion=particion,
            opciones=opciones,
        )
        metadatos.update(
            {
                "estrategia_seleccion": estrategia,
                "metrica_seleccion": "rmse",
                "objetivo_seleccion": "minimizar",
                "folds": len(pliegues) if estrategia == "cv" else None,
                "feature_simple": simple,
                "algoritmos": list(catalogo),
                "configuraciones": deepcopy(catalogo),
                "validacion_externa": particion is not None
                and particion.validacion is not None,
                "validacion_reservada_indices": validacion.tolist()
                if estrategia == "cv"
                else [],
                "test_usado_para_seleccion": False,
                "reajustado_train_completo": estrategia == "cv",
                "reajustado_train_validacion": False,
                "desempate": "primer candidato y algoritmo declarados",
                "muestra_train": len(train),
                "muestra_test": len(test),
                "muestra_validacion": len(validacion) if estrategia == "holdout" else None,
                "muestra_validacion_por_pliegue": [len(v) for _, v in pliegues],
                "pliegues_indices": [v.tolist() for _, v in pliegues],
                "pliegues": [
                    {"train_indices": t.tolist(), "validacion_indices": v.tolist()}
                    for t, v in pliegues
                ],
            }
        )
        filas, ganadores, modelos = [], {}, {}
        for algoritmo, candidatos in catalogo.items():
            features_algoritmo = (
                [simple]
                if algoritmo == "RLS"
                else list(regresor.features)
            )
            for candidato in candidatos:
                fila: dict[str, Any] = {c: None for c in COLUMNAS_TABLA_REGRESION}
                fila.update(
                    algoritmo=algoritmo,
                    configuracion=candidato["nombre"],
                    features=features_algoritmo,
                    parametros=deepcopy(candidato["parametros"]),
                    seleccionado=False,
                    estado="ok",
                    error=None,
                )
                try:
                    features_anteriores = regresor.features
                    regresor.features = features_algoritmo
                    try:
                        metricas_pliegues, modelo, preparados = _evaluar(
                            regresor,
                            base,
                            pliegues,
                            algoritmo,
                            random_state,
                            opciones,
                            candidato["parametros"],
                        )
                    finally:
                        regresor.features = features_anteriores
                    medias = {
                        clave: float(np.mean([m[clave] for m in metricas_pliegues]))
                        for clave in ("rmse", "mae", "mse", "r2")
                    }
                    std_rmse = (
                        float(np.std([m["rmse"] for m in metricas_pliegues], ddof=1))
                        if len(metricas_pliegues) > 1
                        else 0.0
                    )
                    fila.update(
                        {
                            "rmse_validacion": medias["rmse"],
                            "mae_validacion": medias["mae"],
                            "mse_validacion": medias["mse"],
                            "r2_validacion": medias["r2"],
                            "puntuacion_validacion": medias["rmse"],
                            "puntuacion_validacion_std": std_rmse,
                            "puntuaciones_pliegues": [
                                float(m["rmse"]) for m in metricas_pliegues
                            ],
                            "metricas_pliegues": metricas_pliegues,
                            "parametros": parametros_regresion(modelo),
                        }
                    )
                    anterior = ganadores.get(algoritmo)
                    if anterior is None or medias["rmse"] < anterior["puntuacion"]:
                        ganadores[algoritmo] = {
                            "puntuacion": medias["rmse"],
                            "std": std_rmse,
                            "candidato": candidato,
                            "features": features_algoritmo,
                            "fila": len(filas),
                            "modelo": modelo,
                            "datos_preparados": preparados,
                            "metricas_pliegues": metricas_pliegues,
                        }
                except (ValueError, TypeError, RuntimeError, ImportError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    fila.update(estado="error", error=f"{type(exc).__name__}: {exc}")
                filas.append(fila)

        mejor_algoritmo = min(
            ganadores, key=lambda a: ganadores[a]["puntuacion"], default=None
        )
        metadatos["selecciones"] = {
            a: g["candidato"]["nombre"] for a, g in ganadores.items()
        }
        metadatos["mejor_algoritmo_por_validacion"] = mejor_algoritmo
        metadatos["errores_prueba"] = {}
        tabla = pd.DataFrame(filas, columns=COLUMNAS_TABLA_REGRESION)
        for ganador in ganadores.values():
            tabla.loc[ganador["fila"], "seleccionado"] = True
        metadatos["candidatos_exitosos"] = int((tabla["estado"] == "ok").sum())
        metadatos["candidatos_fallidos"] = int((tabla["estado"] == "error").sum())
        mejores = {}
        for algoritmo, ganador in ganadores.items():
            features_anteriores = regresor.features
            regresor.features = ganador["features"]
            try:
                if estrategia == "cv":
                    modelo_final, preparados_final = regresor._ajustar(
                        base,
                        train,
                        algoritmo,
                        random_state,
                        opciones,
                        ganador["candidato"]["parametros"],
                    )
                else:
                    modelo_final, preparados_final = (
                        ganador["modelo"],
                        ganador["datos_preparados"],
                    )
                meta_modelo = {
                    **metadatos,
                    "features": list(ganador["features"]),
                    "conjunto_evaluacion": "test",
                    "configuracion": ganador["candidato"]["nombre"],
                    "puntuacion_validacion": ganador["puntuacion"],
                    "puntuacion_validacion_std": ganador["std"],
                    "metricas_pliegues": ganador["metricas_pliegues"],
                }
                resultado = regresor._resultado(
                    base,
                    train,
                    test,
                    algoritmo,
                    modelo_final,
                    preparados_final,
                    meta_modelo,
                )
                mejores[algoritmo] = resultado
                modelos[algoritmo] = (modelo_final, preparados_final)
            except (ValueError, TypeError, RuntimeError) as exc:
                metadatos["errores_prueba"][algoritmo] = f"{type(exc).__name__}: {exc}"
            finally:
                regresor.features = features_anteriores

        return ResultadoExperimentoRegresion(
            tabla=tabla,
            mejores=mejores,
            mejor_algoritmo=mejor_algoritmo,
            metadatos=metadatos,
        )


def ejecutar_experimento_clasificacion(*args, **kwargs) -> ResultadoExperimentoClasificacion:
    """Función de conveniencia para ejecutar benchmark de clasificación."""
    return ExperimentoClasificacion().ejecutar(*args, **kwargs)


def ejecutar_experimento_regresion(*args, **kwargs) -> ResultadoExperimentoRegresion:
    """Función de conveniencia para ejecutar benchmark de regresión."""
    return ExperimentoRegresion().ejecutar(*args, **kwargs)


def ejecutar_experimento(modelo, *args, **kwargs):
    """Enrutador de benchmark supervisado según el tipo de modelo."""
    if hasattr(modelo, "_datos_clasificacion") or getattr(modelo, "_tipo_target", lambda: "")() == "clasificacion":
        return ejecutar_experimento_clasificacion(modelo, *args, **kwargs)
    return ejecutar_experimento_regresion(modelo, *args, **kwargs)
