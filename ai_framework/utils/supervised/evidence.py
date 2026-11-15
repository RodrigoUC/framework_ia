"""Read and verify experiment evidence without retraining models."""

from __future__ import annotations

import csv as csv_module
import hashlib
import json
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .regression import metricas_regresion
from ..data.csv_config import ConfiguracionCSV
from .regression_config import (
    ALGORITMOS_REGRESION,
    configuraciones_experimento_regresion,
)

METRICAS = ("rmse", "mae", "mse", "r2")
_COLUMNAS_JSON = (
    "features",
    "parametros",
    "puntuaciones_pliegues",
    "metricas_pliegues",
)


@dataclass(frozen=True)
class RegressionEvidence:
    """Validated snapshots to build reports and verify experimental results."""

    datos: pd.DataFrame
    configuracion: dict[str, Any]
    comparacion: pd.DataFrame
    mejores: dict[str, Any]
    predicciones: dict[str, dict[str, pd.DataFrame]]
    procedencia: dict[str, Any]
    hashes: dict[str, str]

    @property
    def data(self) -> pd.DataFrame:
        return self.datos

    @property
    def configuration(self) -> dict[str, Any]:
        return self.configuracion

    @property
    def comparison(self) -> pd.DataFrame:
        return self.comparacion

    @property
    def best(self) -> dict[str, Any]:
        return self.mejores

    @property
    def predictions(self) -> dict[str, dict[str, pd.DataFrame]]:
        return self.predicciones

    @property
    def provenance(self) -> dict[str, Any]:
        return self.procedencia


EvidenciaRegresion = RegressionEvidence
EvidenciaLab03 = EvidenciaRegresion


def _require(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


def _object_hook(pares):
    resultado = {}
    for clave, valor in pares:
        _require(clave not in resultado, f"Clave JSON duplicada: {clave}.")
        resultado[clave] = valor
    return resultado


def _json(texto):
    def constante(valor):
        raise ValueError(f"Valor JSON no finito: {valor}.")

    def finito(valor):
        if isinstance(valor, float):
            _require(np.isfinite(valor), "El JSON contiene valores no finitos.")
        elif isinstance(valor, dict):
            for contenido in valor.values():
                finito(contenido)
        elif isinstance(valor, list):
            for contenido in valor:
                finito(contenido)

    resultado = json.loads(texto, object_pairs_hook=_objeto, parse_constant=constante)
    finito(resultado)
    return resultado


def _normalize(objeto):
    return _json(json.dumps(objeto, allow_nan=False))


def _is_close(actual, esperado, contexto):
    _require(
        np.asarray(actual).dtype.kind in "iuf",
        f"{contexto}: se esperaban valores numéricos reales.",
    )
    valores = np.asarray(actual, dtype=float)
    referencia = np.asarray(esperado, dtype=float)
    _require(np.isfinite(valores).all(), f"{contexto}: valores no finitos.")
    _require(
        valores.shape == referencia.shape
        and np.isfinite(referencia).all()
        and np.allclose(valores, referencia, rtol=1e-9, atol=1e-10),
        f"{contexto}: los valores no coinciden.",
    )


def _validate_indices(valor, contexto):
    _require(isinstance(valor, list), f"{contexto}: se esperaba una lista de índices.")
    _require(
        all(isinstance(i, int) and not isinstance(i, bool) for i in valor),
        f"{contexto}: los índices deben ser enteros.",
    )
    _require(len(set(valor)) == len(valor), f"{contexto}: índices duplicados.")
    return valor


def _validate_partitions(metadata, datos):
    train, test = train_test_split(
        np.arange(len(datos)), test_size=0.25, random_state=42
    )
    train, test = train.tolist(), test.tolist()
    for nombre, esperado in (("train_indices", train), ("test_indices", test)):
        _require(
            _validate_indices(metadata[nombre], nombre) == esperado,
            f"{nombre}: no corresponde a la partición 75/25 con semilla 42.",
        )
    divisor = KFold(n_splits=5, shuffle=True, random_state=42)
    pliegues = [
        {
            "train_indices": [train[i] for i in t],
            "validacion_indices": [train[i] for i in v],
        }
        for t, v in divisor.split(train)
    ]
    _require(
        metadata["pliegues"] == pliegues, "Los pliegues no corresponden a KFold(5)."
    )
    validaciones = [p["validacion_indices"] for p in pliegues]
    _require(
        metadata["pliegues_indices"] == validaciones,
        "Índices de pliegues incoherentes.",
    )
    _require(
        metadata["muestra_train"] == len(train)
        and metadata["muestra_test"] == len(test)
        and metadata["muestra_validacion_por_pliegue"] == list(map(len, validaciones)),
        "Tamaños de partición incoherentes.",
    )
    return train, test


def _validate_protocol(metadata, datos):
    esperado = {
        "semilla": 42,
        "semilla_particion": 42,
        "particion_externa": False,
        "estrategia_seleccion": "cv",
        "metrica_seleccion": "rmse",
        "objetivo_seleccion": "minimizar",
        "folds": 5,
        "validacion_indices": [],
        "validacion_reservada_indices": [],
        "validacion_externa": False,
        "test_usado_para_seleccion": False,
        "reajustado_train_completo": True,
        "reajustado_train_validacion": False,
        "candidatos_exitosos": 14,
        "candidatos_fallidos": 0,
        "errores_prueba": {},
        "conjunto_evaluacion": "test",
    }
    for clave, valor in esperado.items():
        _require(metadata[clave] == valor, f"Protocolo inválido: {clave}.")
    preparacion = metadata["preprocesamiento"]
    _require(
        preparacion["ajustado_solo_train"] is True
        and preparacion["seleccion_k"] is None
        and preparacion["imputar"] is True
        and preparacion["estandarizar"] is True,
        "El preprocesamiento debe declarar ajuste sólo con train, imputación y escala.",
    )
    from ai_framework.data.partition import huella_datos

    _require(
        metadata["huella_datos"] == huella_datos(datos), "Huella de datos incoherente."
    )
    for nombre in ("python", "numpy", "pandas", "scikit-learn"):
        _require(bool(metadata["versiones"][nombre]), f"Falta la versión de {nombre}.")
    target, features, simple = (
        metadata["target"],
        metadata["features"],
        metadata["feature_simple"],
    )
    _require(target in datos, "La variable objetivo no existe en el CSV.")
    _require(
        isinstance(features, list)
        and len(features) >= 2
        and len(set(features)) == len(features)
        and target not in features
        and set(features).issubset(datos.columns)
        and simple in features
        and pd.api.types.is_numeric_dtype(datos[simple]),
        "Predictoras o predictora simple inválidas.",
    )
    _is_close(datos[target], datos[target], "Variable objetivo")
    _require(datos[target].nunique() > 1, "La variable objetivo es constante.")
    return _validate_partitions(metadata, datos)


def _validate_internal_cv(metadata, algoritmo, parametros):
    if algoritmo not in ("LassoCV", "RidgeCV"):
        return
    cv = {"tipo": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42}
    _require(parametros["cv"] == cv, "Configuración CV interna inválida.")
    esperado = {
        **cv,
        "scoring": "mean_squared_error"
        if algoritmo == "LassoCV"
        else "neg_mean_squared_error",
        "preprocesamiento_por_pliegue": False,
        "uso": "seleccion_alpha_solo_train",
    }
    _require(
        metadata["cv_interna"] == esperado, "Limitación de CV interna no documentada."
    )
    _require(
        metadata["alpha_elegido"] in parametros["alphas"],
        "Alpha elegido ajeno a la cuadrícula.",
    )


def _validate_comparison(tabla, metadata):
    catalogo = _normalize(configuraciones_experimento_regresion())
    _require(
        metadata["algoritmos"] == list(ALGORITMOS_REGRESION),
        "Catálogo incompleto de nueve modelos.",
    )
    _require(
        metadata["configuraciones"] == catalogo,
        "El catálogo no contiene los 14 candidatos acordados.",
    )
    _require(len(tabla) == 14, "La comparación debe contener 14 candidatos.")
    for columna in _COLUMNAS_JSON:
        tabla[columna] = tabla[columna].map(_json)
    _require(
        tabla.estado.eq("ok").all() and tabla.error.isna().all(),
        "Existen candidatos fallidos.",
    )
    pares = [(a, c) for a, candidatos in catalogo.items() for c in candidatos]
    ganadores = {}
    for (_, fila), (algoritmo, candidato) in zip(tabla.iterrows(), pares, strict=True):
        _require(
            fila.algoritmo == algoritmo and fila.configuracion == candidato["nombre"],
            "Orden, algoritmo o configuración duplicada/inválida en comparación.",
        )
        features = (
            [metadata["feature_simple"]] if algoritmo == "RLS" else metadata["features"]
        )
        _require(fila.features == features, f"{algoritmo}: predictoras incoherentes.")
        _require(
            isinstance(fila.parametros, dict), "Parámetros del candidato inválidos."
        )
        for clave, valor in candidato["parametros"].items():
            _require(
                fila.parametros[clave] == valor,
                f"{algoritmo}: parámetro {clave} incoherente.",
            )
        if algoritmo in ("Lasso", "LassoCV", "Ridge", "DT", "RF"):
            _require(
                fila.parametros["random_state"] == 42, "Semilla del estimador inválida."
            )
        medidas = fila.metricas_pliegues
        _require(
            isinstance(medidas, list) and len(medidas) == 5,
            "Faltan métricas de pliegues.",
        )
        for numero, medida in enumerate(medidas, start=1):
            _require(medida["pliegue"] == numero, "Número de pliegue incoherente.")
            _validate_internal_cv(medida, algoritmo, fila.parametros)
            _is_close(medida["mse"], medida["rmse"] ** 2, "MSE y RMSE por pliegue")
            _require(
                all(medida[m] >= 0 for m in ("rmse", "mae", "mse"))
                and medida["r2"] <= 1 + 1e-9,
                "Métricas de pliegue fuera de su dominio.",
            )
        for metrica in METRICAS:
            valores = [medida[metrica] for medida in medidas]
            _is_close(valores, valores, f"{algoritmo}: {metrica} por pliegue")
            _is_close(fila[f"{metrica}_validacion"], np.mean(valores), f"Media {metrica}")
        scores = [m["rmse"] for m in medidas]
        _is_close(fila.puntuaciones_pliegues, scores, "Scores por pliegue")
        _is_close(fila.puntuacion_validacion, np.mean(scores), "RMSE de selección")
        _is_close(
            fila.puntuacion_validacion_std,
            np.std(scores, ddof=1),
            "Desviación de validación",
        )
        anterior = ganadores.get(algoritmo)
        if (
            anterior is None
            or fila.puntuacion_validacion < anterior.puntuacion_validacion
        ):
            ganadores[algoritmo] = fila
    for algoritmo, fila in ganadores.items():
        grupo = tabla[tabla.algoritmo == algoritmo]
        _require(
            grupo.seleccionado.tolist() == [i == fila.name for i in grupo.index],
            f"{algoritmo}: selección distinta del primer mínimo de validación.",
        )
    _require(
        metadata["selecciones"] == {a: f.configuracion for a, f in ganadores.items()},
        "Selecciones de configuración incoherentes.",
    )
    mejor = min(ganadores, key=lambda a: ganadores[a].puntuacion_validacion)
    _require(
        metadata["mejor_algoritmo_por_validacion"] == mejor,
        "Ganador global incoherente.",
    )
    return ganadores


def _validate_dataset(datos, csv, procedencia, dataset, contenido):
    digest = hashlib.sha256(contenido).hexdigest()
    for nombre, objeto in (("procedencia", procedencia), ("dataset", dataset)):
        _require(objeto["sha256"] == digest, f"SHA-256 del CSV distinto en {nombre}.")
        _require(objeto["archivo"] == csv.name, f"Nombre del CSV distinto en {nombre}.")
        _require(
            objeto["filas"] == len(datos), f"Cantidad de filas distinta en {nombre}."
        )
        _require(
            objeto["columnas"] == datos.columns.tolist(),
            f"Columnas distintas en {nombre}.",
        )
    for clave in ("separador", "decimal", "encoding"):
        _require(
            procedencia["lectura"][clave] == dataset["lectura"][clave],
            "Lectura del CSV incoherente.",
        )
    _require(
        not procedencia["lectura"].get("usar_primera_columna_como_indice", False),
        "La evidencia académica requiere índices de fila del CSV sin columna índice.",
    )
    _require(
        dataset.get("fuente") == procedencia.get("fuente"),
        "Fuente del dataset incoherente.",
    )


def load_evidence(resultados: Path, csv: Path, procedencia: Path) -> EvidenciaRegresion:
    """Valida CSV, procedencia y los 39 artefactos de un benchmark completo.

    El protocolo académico es semilla 42, prueba 25% y CV externa de cinco
    pliegues. La comprobación recalcula particiones y métricas, nunca modelos.
    Los errores de entrada se comunican como ``ValueError`` con contexto.
    """
    try:
        return _load(Path(resultados), Path(csv), Path(procedencia))
    except (
        OSError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        UnicodeError,
    ) as exc:
        raise ValueError(f"Evidencia Lab03 incompleta o inválida: {exc}") from exc


def _load(resultados, csv, ruta_procedencia):
    hashes = {}

    def _read(ruta, nombre):
        contenido = ruta.read_bytes()
        hashes[nombre] = hashlib.sha256(contenido).hexdigest()
        return contenido

    def _read_json(nombre):
        valor = _json(_read(resultados / nombre, f"resultados/{nombre}"))
        _require(isinstance(valor, dict), f"{nombre}: se esperaba un objeto JSON.")
        return valor

    procedencia = _json(_read(ruta_procedencia, "datos/procedencia.json"))
    configuracion = _read_json("configuracion.json")
    contenido = _read(csv, f"datos/{csv.name}")
    _require(bool(contenido), "El CSV está vacío.")
    lectura = configuracion["dataset"]["lectura"]
    encabezado = next(
        csv_module.reader(
            StringIO(contenido.decode(lectura["encoding"])),
            delimiter=lectura["separador"],
        )
    )
    _require(
        len(set(encabezado)) == len(encabezado), "El CSV tiene columnas duplicadas."
    )
    from ai_framework.data.sources import CargadorCSV
    datos = CargadorCSV.cargar_bytes(contenido, ConfiguracionCSV(**lectura))
    _validate_dataset(datos, csv, procedencia, configuracion["dataset"], contenido)
    original = procedencia.get("original")
    if original is not None:
        nombre = original["archivo"]
        _require(
            Path(nombre).name == nombre and nombre not in ("", ".", ".."),
            "Ruta original inválida.",
        )
        raw = _read(ruta_procedencia.parent / nombre, f"datos/{nombre}")
        _require(
            hashlib.sha256(raw).hexdigest() == original["sha256"],
            "SHA-256 de la fuente original distinto.",
        )
    metadata = configuracion["experimento"]
    train, test = _validate_protocol(metadata, datos)
    argumentos = configuracion["argumentos"]
    for clave, valor in (("semilla", 42), ("folds", 5), ("test_size", 0.25)):
        _require(
            argumentos[clave] == valor, f"Argumento de ejecución inválido: {clave}."
        )
    _require(
        argumentos["target"] == metadata["target"], "Target de ejecución incoherente."
    )
    _require(
        not argumentos.get("sin_escalar", False)
        and not argumentos.get("sin_imputar", False),
        "Los argumentos contradicen el preprocesamiento declarado.",
    )
    _require(
        all(
            configuracion["versiones"].get(nombre) == valor
            for nombre, valor in metadata["versiones"].items()
        ),
        "Versiones de ejecución incoherentes.",
    )
    comparacion = pd.read_csv(
        BytesIO(
            _read(
                resultados / "comparacion_validacion.csv",
                "resultados/comparacion_validacion.csv",
            )
        )
    )
    ganadores = _validate_comparison(comparacion, metadata)
    mejores = _read_json("mejores_modelos.json")
    _require(
        set(mejores) == set(ALGORITMOS_REGRESION), "Faltan modelos en mejores_modelos.json."
    )
    predicciones = {}
    for algoritmo in ALGORITMOS_REGRESION:
        resumen, fila = mejores[algoritmo], ganadores[algoritmo]
        individual = _read_json(f"{algoritmo}/configuracion.json")
        for clave in ("dataset", "argumentos", "versiones"):
            _require(
                individual[clave] == configuracion[clave],
                f"{algoritmo}: contexto individual incoherente.",
            )
        modelo = individual["modelo"]
        _require(modelo["algoritmo"] == algoritmo, "Algoritmo del snapshot incoherente.")
        for clave in (
            "target",
            "features",
            "parametros",
            "metadatos",
            "muestra_train",
            "muestra_test",
        ):
            _require(
                modelo[clave] == resumen[clave],
                f"{algoritmo}: snapshot y resumen distintos en {clave}.",
            )
        _require(
            resumen["target"] == metadata["target"]
            and resumen["features"] == fila.features,
            "Variables del snapshot incoherentes.",
        )
        _require(
            resumen["parametros"] == fila.parametros,
            "Parámetros del ganador incoherentes.",
        )
        contexto = resumen["metadatos"]
        for clave, valor in metadata.items():
            if clave != "features":
                _require(
                    contexto[clave] == valor,
                    f"{algoritmo}: metadato común distinto en {clave}.",
                )
        _require(
            contexto["features"] == fila.features
            and contexto["configuracion"] == fila.configuracion,
            "Configuración del ganador incoherente.",
        )
        _is_close(
            contexto["puntuacion_validacion"],
            fila.puntuacion_validacion,
            "Score del ganador",
        )
        _is_close(
            contexto["puntuacion_validacion_std"],
            fila.puntuacion_validacion_std,
            "Desviación del ganador",
        )
        _validate_internal_cv(contexto, algoritmo, fila.parametros)
        metricas = _read_json(f"{algoritmo}/metricas.json")
        predicciones[algoritmo] = {}
        for conjunto, indices in (("train", train), ("test", test)):
            nombre = f"{algoritmo}/predicciones_{conjunto}.csv"
            pred = pd.read_csv(
                BytesIO(_read(resultados / nombre, f"resultados/{nombre}")),
                index_col="fila",
            )
            _require(
                pred.index.is_unique
                and pd.api.types.is_integer_dtype(pred.index.dtype)
                and pred.index.tolist() == indices,
                f"{algoritmo}/{conjunto}: índices inválidos.",
            )
            _require(
                pred.columns.tolist() == ["real", "prediccion", "residuo"],
                "Columnas de predicción inválidas.",
            )
            _is_close(pred.real, datos.loc[indices, metadata["target"]], "Valores reales")
            _is_close(pred.residuo, pred.real - pred.prediccion, "Residuos")
            calculadas = metricas_regresion(pred.real, pred.prediccion)
            _require(set(metricas[conjunto]) == set(METRICAS), "Métricas incompletas.")
            _require(
                metricas[conjunto] == resumen[f"metricas_{conjunto}"],
                "Métricas del snapshot y resumen distintas.",
            )
            _require(
                resumen[f"muestra_{conjunto}"] == len(indices),
                "Tamaño del snapshot incoherente.",
            )
            for metrica in METRICAS:
                _is_close(
                    metricas[conjunto][metrica],
                    calculadas[metrica],
                    f"{algoritmo}/{conjunto}/{metrica}",
                )
            predicciones[algoritmo][conjunto] = pred
    return EvidenciaRegresion(
        datos, configuracion, comparacion, mejores, predicciones, procedencia, hashes
    )

# Backward compatibility aliases
cargar_evidencia = load_evidence

# Backward compatibility aliases
_exigir = _require
_objeto = _object_hook
_normalizar = _normalize
_cerca = _is_close
_indices = _validate_indices
_particiones = _validate_partitions
_protocolo = _validate_protocol
_cv_interna = _validate_internal_cv
_comparacion = _validate_comparison
_validar_dataset = _validate_dataset
_cargar = _load
