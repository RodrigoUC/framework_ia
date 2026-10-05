"""Ejecuta clasificación reproducible sobre un CSV, sin acoplarse a Streamlit.

Las salidas son evidencia experimental, no un informe ni una recomendación
médica o de seguridad del agua. Nunca se descarga ni publica el dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from framework_ia.datos.fuentes import CargadorCSV
from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.utils import ConfiguracionCSV


def _json_seguro(valor: Any) -> Any:
    """Convierte tipos científicos sin introducir NaN/Infinity en el JSON."""
    if isinstance(valor, dict):
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


def _guardar_json(ruta: Path, contenido: Any) -> None:
    ruta.write_text(
        json.dumps(
            _json_seguro(contenido), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LAB 02: comparar clasificadores con validación y test reservado."
    )
    parser.add_argument(
        "csv", type=Path, help="CSV local: no se descarga ni transmite."
    )
    parser.add_argument(
        "--target", required=True, help="Columna de clase, por ejemplo Potability."
    )
    parser.add_argument(
        "--features", nargs="+", help="Predictoras; por defecto todas menos target."
    )
    parser.add_argument("--salida", type=Path, default=Path("salida_lab02"))
    parser.add_argument("--separador", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fracción del train restante reservada para validación.",
    )
    parser.add_argument(
        "--algoritmos", nargs="+", choices=["KNN", "DT", "RF", "XGBoost", "AdaBoost"]
    )
    parser.add_argument("--incluir-categoricas", action="store_true")
    parser.add_argument("--sin-escalar", action="store_true")
    parser.add_argument("--sin-imputar", action="store_true")
    parser.add_argument(
        "--fuente", help="URL o referencia de procedencia; sólo se registra localmente."
    )
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    """Lee el CSV crudo y exporta sólo después de completar el experimento."""
    if argumentos.salida.exists() and any(argumentos.salida.iterdir()):
        raise ValueError(
            "La carpeta de salida contiene archivos; elija una carpeta nueva."
        )
    contenido = argumentos.csv.read_bytes()
    configuracion_csv = ConfiguracionCSV(
        separador=argumentos.separador,
        decimal=argumentos.decimal,
        encoding=argumentos.encoding,
    )
    datos = CargadorCSV.cargar_bytes(contenido, configuracion_csv)
    clasificador = Clasificacion(
        dataframe=datos, target=argumentos.target, features=argumentos.features
    )
    experimento = clasificador.experimentar(
        algoritmos=argumentos.algoritmos,
        test_size=argumentos.test_size,
        validation_size=argumentos.validation_size,
        random_state=argumentos.semilla,
        incluir_categoricas=argumentos.incluir_categoricas,
        imputar=not argumentos.sin_imputar,
        estandarizar=not argumentos.sin_escalar,
    )
    if not experimento.mejores:
        errores = experimento.tabla["error"].dropna().tolist()
        errores.extend(experimento.metadatos.get("errores_prueba", {}).values())
        raise ValueError(
            f"Ningún modelo pudo evaluarse. Revise las configuraciones: {errores}"
        )
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    paquetes = ["numpy", "pandas", "scikit-learn", "xgboost"]
    versiones = {}
    for paquete in paquetes:
        try:
            versiones[paquete] = importlib.metadata.version(paquete)
        except importlib.metadata.PackageNotFoundError:
            versiones[paquete] = None
    _guardar_json(
        argumentos.salida / "configuracion.json",
        {
            "argumentos": vars(argumentos),
            "dataset": {
                "archivo": argumentos.csv.name,
                "sha256": hashlib.sha256(contenido).hexdigest(),
                "filas": len(datos),
                "columnas": list(datos.columns),
                "fuente": argumentos.fuente,
            },
            "versiones": versiones,
            "experimento": experimento.metadatos,
            "advertencia": "Resultados educativos; requieren interpretación y no validan uso clínico ni potabilidad real.",
        },
    )
    tabla = experimento.tabla.copy()
    for columna in tabla.columns:
        if tabla[columna].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            tabla[columna] = tabla[columna].map(
                lambda v: (
                    json.dumps(_json_seguro(v), ensure_ascii=False, allow_nan=False)
                    if isinstance(v, (dict, list, tuple))
                    else v
                )
            )
    tabla.to_csv(argumentos.salida / "comparacion_validacion.csv", index=False)
    resumen = {}
    for algoritmo, resultado in experimento.mejores.items():
        directorio = argumentos.salida / algoritmo
        directorio.mkdir()
        resultado.matriz_confusion.to_csv(directorio / "matriz_confusion_test.csv")
        pd.concat(
            [resultado.y_true.rename("real"), resultado.y_pred.rename("prediccion")],
            axis=1,
        ).to_csv(directorio / "predicciones_test.csv", index_label="fila")
        resumen[algoritmo] = {
            "parametros": resultado.parametros,
            "metricas_test": resultado.metricas,
            "metadatos": resultado.metadatos,
            "muestra_train": resultado.muestra_train,
            "muestra_test": resultado.muestra_test,
        }
    _guardar_json(argumentos.salida / "mejores_modelos.json", resumen)
    return argumentos.salida


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        salida = ejecutar(argumentos)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"No se pudo ejecutar LAB 02: {exc}\n")
    configuracion = json.loads(
        (salida / "configuracion.json").read_text(encoding="utf-8")
    )
    metadatos = configuracion["experimento"]
    fallidos = metadatos.get("candidatos_fallidos", 0)
    errores_test = metadatos.get("errores_prueba", {})
    print(f"Resultados guardados en {salida}. Revise validación y test por separado.")
    if fallidos or errores_test:
        print(
            f"Ejecución parcial: {fallidos} candidatos fallidos; errores de test: {errores_test}. Revise los archivos de salida."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
