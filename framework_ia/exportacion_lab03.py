"""Evidencia de regresión compartida por CLI e interfaz, sin entrenamiento ni E/S."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .exportacion_lab02 import _bytes_json, _json_seguro, construir_zip_experimento
from .resultados import ResultadoExperimentoRegresion, ResultadoRegresion

__all__ = [
    "construir_archivos_experimento",
    "construir_archivos_modelo",
    "construir_zip_experimento",
]


def _configuracion(resultado, argumentos, dataset, versiones):
    return {
        "argumentos": argumentos,
        "dataset": dataset,
        "versiones": {**(versiones or {}), **resultado.metadatos.get("versiones", {})},
        "advertencia": "Resultados educativos; no constituyen una recomendación clínica.",
    }


def construir_archivos_modelo(
    resultado: ResultadoRegresion,
    *,
    argumentos: Mapping[str, Any],
    dataset: Mapping[str, Any],
    versiones: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Exporta una ejecución individual sin crear un experimento artificial."""
    configuracion = _configuracion(resultado, argumentos, dataset, versiones)
    configuracion["modelo"] = {
        "algoritmo": resultado.algoritmo,
        "target": resultado.target,
        "features": resultado.features,
        "parametros": resultado.parametros,
        "metadatos": resultado.metadatos,
        "muestra_train": resultado.muestra_train,
        "muestra_test": resultado.muestra_test,
    }
    archivos = {
        "configuracion.json": _bytes_json(configuracion),
        "metricas.json": _bytes_json(
            {"train": resultado.metricas_train, "test": resultado.metricas}
        ),
    }
    for conjunto, real, prediccion, residuo in (
        ("train", resultado.y_train, resultado.y_pred_train, resultado.residuos_train),
        ("test", resultado.y_true, resultado.y_pred, resultado.residuos),
    ):
        predicciones = pd.concat(
            [
                real.rename("real"),
                prediccion.rename("prediccion"),
                residuo.rename("residuo"),
            ],
            axis=1,
        )
        archivos[f"predicciones_{conjunto}.csv"] = predicciones.to_csv(
            index_label="fila", lineterminator="\n"
        ).encode("utf-8")
    return archivos


def construir_archivos_experimento(
    resultado: ResultadoExperimentoRegresion,
    *,
    argumentos: Mapping[str, Any],
    dataset: Mapping[str, Any],
    versiones: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Serializa snapshots con rutas relativas y versiones del ajuste prioritarias.

    ``dataset`` conserva la procedencia y SHA-256 de los bytes originales,
    no una huella de un CSV reconstruido. La misma función produce los bytes
    que la CLI escribe y la interfaz ofrece para descargar.
    """
    configuracion = _configuracion(resultado, argumentos, dataset, versiones)
    configuracion["experimento"] = resultado.metadatos
    archivos = {"configuracion.json": _bytes_json(configuracion)}
    tabla = resultado.tabla.copy(deep=True)
    for columna in tabla.columns:
        tabla[columna] = tabla[columna].map(
            lambda valor: (
                json.dumps(_json_seguro(valor), ensure_ascii=False, allow_nan=False)
                if isinstance(valor, (dict, list, tuple))
                else valor
            )
        )
    archivos["comparacion_validacion.csv"] = tabla.to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    resumen = {}
    for algoritmo, ganador in resultado.mejores.items():
        archivos_modelo = construir_archivos_modelo(
            ganador, argumentos=argumentos, dataset=dataset, versiones=versiones
        )
        archivos.update(
            {
                f"{algoritmo}/{nombre}": contenido
                for nombre, contenido in archivos_modelo.items()
            }
        )
        resumen[algoritmo] = {
            "target": ganador.target,
            "features": ganador.features,
            "parametros": ganador.parametros,
            "metricas_train": ganador.metricas_train,
            "metricas_test": ganador.metricas,
            "metadatos": ganador.metadatos,
            "muestra_train": ganador.muestra_train,
            "muestra_test": ganador.muestra_test,
        }
    archivos["mejores_modelos.json"] = _bytes_json(resumen)
    return archivos
