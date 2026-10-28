"""Run reproducible regression benchmarks on local CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ai_framework.data.sources import CargadorCSV as CSVLoader
from ai_framework.models.supervised import Regresion as Regression
from ai_framework.utils.data import ConfiguracionCSV as CSVConfig
from ai_framework.utils.supervised.regression_config import ALGORITMOS_LAB3
from ai_framework.utils.supervised.export import (
    construir_archivos_experimento as build_experiment_files,
)

__all__ = [
    "create_parser",
    "crear_parser",
    "run",
    "ejecutar",
    "main",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare regressors by RMSE with reserved holdout test."
    )
    parser.add_argument(
        "csv", type=Path, help="Local CSV file path."
    )
    parser.add_argument(
        "--target", required=True, help="Target real numeric column name."
    )
    parser.add_argument(
        "--features", nargs="+", help="Predictor features; default all except target."
    )
    parser.add_argument(
        "--feature-simple",
        help="Single predictor for SLR; default first numeric feature selected.",
    )
    parser.add_argument("--salida", "--output", dest="salida", type=Path, default=Path("salida_regresion"))
    parser.add_argument("--separador", "--separator", dest="separador", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--semilla", "--seed", dest="semilla", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of train reserved for validation when holdout strategy is used.",
    )
    parser.add_argument("--algoritmos", "--algorithms", dest="algoritmos", nargs="+", choices=ALGORITMOS_LAB3)
    parser.add_argument(
        "--estrategia",
        "--strategy",
        dest="estrategia",
        choices=("cv", "holdout"),
        default=None,
        help="Default CV over train; holdout reserves a single validation set.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--incluir-categoricas", "--include-categorical", dest="incluir_categoricas", action="store_true")
    parser.add_argument("--sin-escalar", "--no-scaling", dest="sin_escalar", action="store_true")
    parser.add_argument("--sin-imputar", "--no-impute", dest="sin_imputar", action="store_true")
    parser.add_argument(
        "--fuente", "--source", dest="fuente", help="Provenance reference URL or identifier; recorded locally only."
    )
    return parser


crear_parser = create_parser


def run(argumentos: argparse.Namespace) -> Path:
    """Validate and run experiments before creating output artifacts."""
    if argumentos.salida.exists() and any(argumentos.salida.iterdir()):
        raise ValueError("La carpeta de salida contiene archivos; elija una carpeta nueva. / Output directory not empty.")
    contenido = argumentos.csv.read_bytes()
    configuracion = CSVConfig(
        separador=argumentos.separador,
        decimal=argumentos.decimal,
        encoding=argumentos.encoding,
    )
    datos = CSVLoader.cargar_bytes(contenido, configuracion)
    experimento = Regression(
        dataframe=datos, target=argumentos.target, features=argumentos.features
    ).experimentar(
        algoritmos=argumentos.algoritmos,
        feature_simple=argumentos.feature_simple,
        test_size=argumentos.test_size,
        validation_size=argumentos.validation_size,
        random_state=argumentos.semilla,
        estrategia=argumentos.estrategia,
        folds=argumentos.folds,
        incluir_categoricas=argumentos.incluir_categoricas,
        imputar=not argumentos.sin_imputar,
        estandarizar=not argumentos.sin_escalar,
    )
    if not experimento.mejores:
        errores = experimento.tabla["error"].dropna().tolist()
        errores.extend(experimento.metadatos["errores_prueba"].values())
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
            "lectura": {
                "separador": argumentos.separador,
                "decimal": argumentos.decimal,
                "encoding": argumentos.encoding,
            },
        },
    )
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    for nombre, datos_archivo in archivos.items():
        destino = argumentos.salida / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(datos_archivo)
    return argumentos.salida


ejecutar = run


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    argumentos = parser.parse_args(argv)
    try:
        salida = run(argumentos)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"Could not run regression: {exc}\n")
    metadata = json.loads((salida / "configuracion.json").read_text(encoding="utf-8"))[
        "experimento"
    ]
    print(f"Resultados guardados en {salida}. RMSE de validación selecciona; prueba sólo evalúa. / Results saved.")
    if metadata["candidatos_fallidos"] or metadata["errores_prueba"]:
        print(f"Ejecución parcial: {metadata['candidatos_fallidos']} candidatos fallidos; errores de prueba: {metadata['errores_prueba']}. Revise la evidencia exportada. / Partial execution.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

Regresion = Regression
