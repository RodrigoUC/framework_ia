"""Run reproducible classification benchmarks on local CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ai_framework.data.sources import CargadorCSV as CSVLoader
from ai_framework.models.supervised import Clasificacion as Classification
from ai_framework.utils.data import ConfiguracionCSV as CSVConfig
from ai_framework.utils.supervised.export import (
    _json_seguro,
    construir_archivos_experimento as build_experiment_files,
)

__all__ = [
    "_json_seguro",
    "create_parser",
    "crear_parser",
    "ejecutar",
    "main",
    "run",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare classifiers with validation and holdout test set."
    )
    parser.add_argument(
        "csv", type=Path, help="Local CSV file path."
    )
    parser.add_argument(
        "--target", required=True, help="Target class column name (e.g. Potability)."
    )
    parser.add_argument(
        "--features", nargs="+", help="Predictor features; default all except target."
    )
    parser.add_argument("--salida", "--output", dest="salida", type=Path, default=Path("salida_clasificacion"))
    parser.add_argument("--separador", "--separator", dest="separador", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--semilla", "--seed", dest="semilla", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of remaining train set reserved for validation.",
    )
    parser.add_argument(
        "--algoritmos", "--algorithms", dest="algoritmos", nargs="+", choices=["KNN", "DT", "RF", "XGBoost", "AdaBoost"]
    )
    parser.add_argument(
        "--estrategia",
        "--strategy",
        dest="estrategia",
        choices=["cv", "holdout"],
        default="cv",
        help="cv: stratified cross-validation on train (default); holdout: single validation split.",
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--seleccion-k",
        "--select-k",
        dest="seleccion_k",
        default="auto",
        help="Features to keep in feature selection stage; 'auto' keeps half, 0 disables.",
    )
    parser.add_argument("--incluir-categoricas", "--include-categorical", dest="incluir_categoricas", action="store_true")
    parser.add_argument("--sin-escalar", "--no-scaling", dest="sin_escalar", action="store_true")
    parser.add_argument("--sin-imputar", "--no-impute", dest="sin_imputar", action="store_true")
    parser.add_argument(
        "--fuente", "--source", dest="fuente", help="URL or provenance reference; recorded locally only."
    )
    return parser


crear_parser = create_parser


def run(argumentos: argparse.Namespace) -> Path:
    """Read raw CSV and export artifacts only after experiment finishes."""
    if argumentos.salida.exists() and any(argumentos.salida.iterdir()):
        raise ValueError("La carpeta de salida contiene archivos; elija una carpeta nueva. / Output directory not empty.")
    contenido = argumentos.csv.read_bytes()
    configuracion_csv = CSVConfig(
        separador=argumentos.separador,
        decimal=argumentos.decimal,
        encoding=argumentos.encoding,
    )
    datos = CSVLoader.cargar_bytes(contenido, configuracion_csv)
    clasificador = Classification(
        dataframe=datos, target=argumentos.target, features=argumentos.features
    )
    estrategia = getattr(argumentos, "estrategia", "holdout")
    opciones_estrategia = (
        {
            "estrategia": "cv",
            "folds": argumentos.folds,
            "seleccion_k": "auto"
            if str(argumentos.seleccion_k) == "auto"
            else int(argumentos.seleccion_k),
        }
        if estrategia == "cv"
        else {"validation_size": argumentos.validation_size}
    )
    experimento = clasificador.experimentar(
        algoritmos=argumentos.algoritmos,
        test_size=argumentos.test_size,
        random_state=argumentos.semilla,
        incluir_categoricas=argumentos.incluir_categoricas,
        imputar=not argumentos.sin_imputar,
        estandarizar=not argumentos.sin_escalar,
        **opciones_estrategia,
    )
    if not experimento.mejores:
        errores = experimento.tabla["error"].dropna().tolist()
        errores.extend(experimento.metadatos.get("errores_prueba", {}).values())
        raise ValueError(
            f"No models could be evaluated. Review settings: {errores}"
        )
    archivos = build_experiment_files(
        experimento,
        argumentos=vars(argumentos),
        dataset={
            "archivo": argumentos.csv.name,
            "sha256": hashlib.sha256(contenido).hexdigest(),
            "filas": len(datos),
            "columnas": list(datos.columns),
            "fuente": argumentos.fuente,
        },
    )
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    for nombre, contenido_archivo in archivos.items():
        destino = argumentos.salida / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido_archivo)
    return argumentos.salida


ejecutar = run


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    argumentos = parser.parse_args(argv)
    try:
        salida = run(argumentos)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"Could not run classification: {exc}\n")
    configuracion = json.loads(
        (salida / "configuracion.json").read_text(encoding="utf-8")
    )
    metadatos = configuracion["experimento"]
    fallidos = metadatos.get("candidatos_fallidos", 0)
    errores_test = metadatos.get("errores_prueba", {})
    print(f"Resultados guardados en {salida}. Revise validación y test por separado. / Results saved.")
    if fallidos or errores_test:
        print(f"Ejecución parcial: {fallidos} candidatos fallidos; errores de test: {errores_test}. Revise los archivos de salida. / Partial execution.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

Clasificacion = Classification
