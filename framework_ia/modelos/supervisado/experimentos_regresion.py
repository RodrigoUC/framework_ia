"""Benchmark de regresión: validación común, RMSE mínimo y prueba reservada."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ...resultados import ResultadoExperimentoRegresion
from ...utils.configuracion_regresion import (
    ALGORITMOS_LAB3,
    configuraciones_experimento_regresion,
)
from .algoritmos_regresion import (
    metadatos_regresion,
    normalizar_algoritmo,
    parametros_regresion,
)
from ...datos.particion import dividir_indices
from .regresion import metricas_regresion

COLUMNAS_TABLA = (
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
ERRORES_CANDIDATO = (
    ValueError,
    TypeError,
    RuntimeError,
    ImportError,
    ArithmeticError,
    np.linalg.LinAlgError,
)


def _candidatos(algoritmos, configuraciones):
    if isinstance(algoritmos, str):
        algoritmos = [algoritmos]
    elegidos = list(
        dict.fromkeys(
            normalizar_algoritmo(a)
            for a in (ALGORITMOS_LAB3 if algoritmos is None else algoritmos)
        )
    )
    if not elegidos:
        raise ValueError("Seleccione al menos un algoritmo para experimentar.")
    catalogo = configuraciones_experimento_regresion()
    if configuraciones is not None:
        if not isinstance(configuraciones, dict):
            raise TypeError("configuraciones debe ser un diccionario por algoritmo.")
        vistos = set()
        for clave, candidatos in configuraciones.items():
            codigo = normalizar_algoritmo(clave)
            if codigo in vistos:
                raise ValueError(
                    f"Configuraciones repetidas para {codigo} mediante aliases."
                )
            vistos.add(codigo)
            catalogo[codigo] = candidatos
    resultado = {}
    for algoritmo in elegidos:
        candidatos = catalogo[algoritmo]
        if not isinstance(candidatos, (list, tuple)) or not candidatos:
            raise ValueError(f"{algoritmo} necesita al menos una configuración.")
        nombres = set()
        resultado[algoritmo] = []
        for candidato in candidatos:
            if not isinstance(candidato, dict) or set(candidato) != {
                "nombre",
                "parametros",
            }:
                raise ValueError(
                    "Cada configuración necesita exactamente nombre y parametros."
                )
            nombre, parametros = candidato["nombre"], candidato["parametros"]
            if not isinstance(nombre, str) or not nombre.strip() or nombre in nombres:
                raise ValueError(
                    f"Nombres de configuración vacíos o repetidos para {algoritmo}."
                )
            if not isinstance(parametros, dict) or not all(
                isinstance(k, str) for k in parametros
            ):
                raise ValueError(
                    "parametros debe contener argumentos del estimador por nombre."
                )
            nombres.add(nombre)
            resultado[algoritmo].append(deepcopy(candidato))
    return resultado


def _predictora_simple(regresor, base, feature_simple):
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


def _evaluar(regresor, base, pliegues, algoritmo, random_state, opciones, parametros):
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


def ejecutar_experimento(
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
    """Evalúa todos los candidatos antes de tocar prueba; conserva sus fallos."""
    base = regresor._datos_regresion()
    catalogo = _candidatos(algoritmos, configuraciones)
    simple = _predictora_simple(regresor, base, feature_simple)
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
        features = (
            ([simple] if simple is not None else [])
            if algoritmo == "RLS"
            else list(regresor.features)
        )
        modelo_familia = type(regresor)(
            dataframe=regresor.datos, target=regresor.target, features=features
        )
        modelos[algoritmo] = modelo_familia
        for candidato in candidatos:
            fila = dict.fromkeys(COLUMNAS_TABLA)
            fila.update(
                algoritmo=algoritmo,
                configuracion=candidato["nombre"],
                features=list(features),
                parametros=deepcopy(candidato["parametros"]),
                seleccionado=False,
                estado="ok",
            )
            try:
                medidas, modelo, preparados = _evaluar(
                    modelo_familia,
                    base,
                    pliegues,
                    algoritmo,
                    random_state,
                    opciones,
                    candidato["parametros"],
                )
                puntuaciones = [m["rmse"] for m in medidas]
                for metrica in ("rmse", "mae", "mse", "r2"):
                    fila[f"{metrica}_validacion"] = float(
                        np.mean([m[metrica] for m in medidas])
                    )
                fila.update(
                    parametros=parametros_regresion(modelo),
                    metricas_pliegues=medidas,
                    puntuacion_validacion=float(np.mean(puntuaciones)),
                    puntuacion_validacion_std=float(np.std(puntuaciones, ddof=1))
                    if len(puntuaciones) > 1
                    else 0.0,
                    puntuaciones_pliegues=puntuaciones,
                )
                anterior = ganadores.get(algoritmo)
                if (
                    anterior is None
                    or fila["puntuacion_validacion"] < anterior["score"]
                ):
                    ganadores[algoritmo] = {
                        "score": fila["puntuacion_validacion"],
                        "candidato": candidato,
                        "fila": len(filas),
                        "modelo": modelo,
                        "preparados": preparados,
                    }
            except ERRORES_CANDIDATO as exc:
                fila.update(estado="error", error=f"{type(exc).__name__}: {exc}")
            filas.append(fila)


    mejor_algoritmo = min(ganadores, key=lambda a: ganadores[a]["score"], default=None)
    metadatos.update(
        {
            "selecciones": {a: g["candidato"]["nombre"] for a, g in ganadores.items()},
            "mejor_algoritmo_por_validacion": mejor_algoritmo,
            "errores_prueba": {},
            "candidatos_exitosos": sum(f["estado"] == "ok" for f in filas),
            "candidatos_fallidos": sum(f["estado"] == "error" for f in filas),
        }
    )
    mejores = {}
    for algoritmo, ganador in ganadores.items():
        filas[ganador["fila"]]["seleccionado"] = True
        modelo_familia = modelos[algoritmo]
        try:
            modelo, preparados = ganador["modelo"], ganador["preparados"]
            if estrategia == "cv":
                modelo, preparados = modelo_familia._ajustar(
                    base,
                    train,
                    algoritmo,
                    random_state,
                    opciones,
                    ganador["candidato"]["parametros"],
                )
            contexto = {
                **metadatos,
                "features": list(modelo_familia.features),
                "conjunto_evaluacion": "test",
                "configuracion": ganador["candidato"]["nombre"],
                "puntuacion_validacion": ganador["score"],
                "puntuacion_validacion_std": filas[ganador["fila"]][
                    "puntuacion_validacion_std"
                ],
            }
            mejores[algoritmo] = modelo_familia._resultado(
                base, train, test, algoritmo, modelo, preparados, contexto
            )
        except ERRORES_CANDIDATO as exc:
            metadatos["errores_prueba"][algoritmo] = f"{type(exc).__name__}: {exc}"
    return ResultadoExperimentoRegresion(
        tabla=pd.DataFrame(filas, columns=COLUMNAS_TABLA),
        mejores=mejores,
        mejor_algoritmo=mejor_algoritmo,
        metadatos=metadatos,
    )
