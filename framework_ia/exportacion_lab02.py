"""Archivos de evidencia de clasificación compartidos por la CLI y la interfaz.

La construcción y el ZIP trabajan sólo en memoria. Los llamadores conservan
la procedencia y la huella de los bytes originales del CSV en ``dataset``;
ninguna función descarga datos, entrena modelos ni escribe al disco.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from .resultados import ResultadoExperimentoClasificacion


def _json_seguro(valor: Any) -> Any:
    """Convierte tipos científicos sin introducir NaN/Infinity en JSON."""
    if isinstance(valor, Mapping):
        return {str(k): _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray, pd.Index)):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, (np.integer, np.floating, np.bool_)):
        return _json_seguro(valor.item())
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, (pd.Timestamp, pd.Timedelta)):
        return str(valor)
    return valor


def _bytes_json(contenido: Any) -> bytes:
    texto = json.dumps(
        _json_seguro(contenido), ensure_ascii=False, indent=2, allow_nan=False
    )
    return (texto + "\n").encode("utf-8")


def _versiones(
    resultado: ResultadoExperimentoClasificacion,
    versiones: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prefiere las versiones guardadas al ajustar frente al ambiente actual."""
    registradas = dict(
        resultado.metadatos.get("versiones", {}) if versiones is None else versiones
    )
    for paquete in ("numpy", "pandas", "scikit-learn", "xgboost"):
        if paquete not in registradas:
            try:
                registradas[paquete] = version(paquete)
            except PackageNotFoundError:
                registradas[paquete] = None
    registradas.setdefault("python", platform.python_version())
    return registradas


def construir_archivos_experimento(
    resultado: ResultadoExperimentoClasificacion,
    *,
    argumentos: Mapping[str, Any],
    dataset: Mapping[str, Any],
    versiones: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Serializa un experimento sin modificar sus tablas ni sus metadatos."""
    archivos = {
        "configuracion.json": _bytes_json(
            {
                "argumentos": argumentos,
                "dataset": dataset,
                "versiones": _versiones(resultado, versiones),
                "experimento": resultado.metadatos,
                "advertencia": "Resultados educativos; requieren interpretación y no validan uso clínico ni potabilidad real.",
            }
        )
    }
    tabla = resultado.tabla.copy()
    for columna in tabla.columns:
        if tabla[columna].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            tabla[columna] = tabla[columna].map(
                lambda v: (
                    json.dumps(_json_seguro(v), ensure_ascii=False, allow_nan=False)
                    if isinstance(v, (dict, list, tuple))
                    else v
                )
            )
    archivos["comparacion_validacion.csv"] = tabla.to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    resumen = {}
    for algoritmo, ganador in resultado.mejores.items():
        archivos[f"{algoritmo}/matriz_confusion_test.csv"] = (
            ganador.matriz_confusion.to_csv(lineterminator="\n").encode("utf-8")
        )
        archivos[f"{algoritmo}/predicciones_test.csv"] = (
            pd.concat(
                [ganador.y_true.rename("real"), ganador.y_pred.rename("prediccion")],
                axis=1,
            )
            .to_csv(index_label="fila", lineterminator="\n")
            .encode("utf-8")
        )
        archivos.update(_archivos_analisis(algoritmo, ganador.metadatos.get("analisis")))
        resumen[algoritmo] = {
            "parametros": ganador.parametros,
            "metricas_test": ganador.metricas,
            "metadatos": ganador.metadatos,
            "muestra_train": ganador.muestra_train,
            "muestra_test": ganador.muestra_test,
        }
    archivos["mejores_modelos.json"] = _bytes_json(resumen)
    ranking = resultado.metadatos.get("seleccion_atributos", {}).get(
        "ranking_informacion_mutua_train"
    )
    if ranking:
        archivos["ranking_atributos.csv"] = (
            pd.Series(ranking, name="informacion_mutua")
            .rename_axis("atributo")
            .to_csv(lineterminator="\n")
            .encode("utf-8")
        )
    return archivos


def _archivos_analisis(algoritmo: str, analisis: Mapping[str, Any] | None) -> dict[str, bytes]:
    """Tablas interpretativas del ganador (sólo existen en la estrategia CV)."""
    if not analisis:
        return {}
    archivos = {}
    tablas = {
        "importancia_permutacion.csv": analisis.get("importancia_permutacion"),
        "importancia_modelo.csv": analisis.get("importancia_modelo"),
    }
    for nombre, filas in tablas.items():
        if filas:
            archivos[f"{algoritmo}/{nombre}"] = (
                pd.DataFrame(filas).to_csv(index=False, lineterminator="\n").encode("utf-8")
            )
    roc = analisis.get("roc") or {}
    if "fpr" in roc:
        archivos[f"{algoritmo}/roc_test.csv"] = (
            pd.DataFrame({"fpr": roc["fpr"], "tpr": roc["tpr"]})
            .to_csv(index=False, lineterminator="\n")
            .encode("utf-8")
        )
    arbol = analisis.get("arbol")
    if arbol:
        archivos[f"{algoritmo}/arbol.txt"] = arbol["texto"].encode("utf-8")
    return archivos


def construir_zip_experimento(archivos: Mapping[str, bytes]) -> bytes:
    """Agrupa los mismos archivos sin escribir al disco ni volver a entrenar."""
    memoria = BytesIO()
    with ZipFile(memoria, "w", compression=ZIP_DEFLATED, compresslevel=9) as salida:
        for nombre in sorted(archivos):
            ruta = PurePosixPath(nombre)
            if (
                not nombre
                or ruta.is_absolute()
                or ".." in ruta.parts
                or "\\" in nombre
                or ":" in nombre
                or str(ruta) != nombre
            ):
                raise ValueError(f"Nombre de archivo relativo inválido: {nombre!r}.")
            informacion = ZipInfo(nombre, date_time=(1980, 1, 1, 0, 0, 0))
            informacion.compress_type = ZIP_DEFLATED
            informacion.create_system = 3
            informacion.external_attr = 0o100644 << 16
            salida.writestr(informacion, archivos[nombre], compresslevel=9)
    return memoria.getvalue()
