"""Ejecuta benchmarking de regresión sobre un CSV local, sin interfaz ni red."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from framework_ia.datos.fuentes import CargadorCSV
from framework_ia.exportacion_lab03 import construir_archivos_experimento
from framework_ia.modelos.supervisado import Regresion
from framework_ia.utils import ALGORITMOS_LAB3, ConfiguracionCSV


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LAB 03: comparar regresores por RMSE con prueba reservada."
    )
    parser.add_argument(
        "csv", type=Path, help="CSV local: no se descarga ni transmite."
    )
    parser.add_argument(
        "--target", required=True, help="Columna objetivo numérica real."
    )
    parser.add_argument(
        "--features", nargs="+", help="Predictoras; por defecto todas menos target."
    )
    parser.add_argument(
        "--feature-simple",
        help="Predictora de RLS; por defecto la primera numérica seleccionada.",
    )
    parser.add_argument("--salida", type=Path, default=Path("salida_lab03"))
    parser.add_argument("--separador", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fracción de train para validación cuando se usa holdout.",
    )
    parser.add_argument("--algoritmos", nargs="+", choices=ALGORITMOS_LAB3)
    parser.add_argument(
        "--estrategia",
        choices=("cv", "holdout"),
        default=None,
        help="Por defecto CV sobre train; holdout reserva una validación.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--incluir-categoricas", action="store_true")
    parser.add_argument("--sin-escalar", action="store_true")
    parser.add_argument("--sin-imputar", action="store_true")
    parser.add_argument(
        "--fuente", help="Referencia de procedencia; sólo se registra localmente."
    )
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    """Valida y experimenta antes de crear archivos; no sobrescribe evidencia."""
    if argumentos.salida.exists() and any(argumentos.salida.iterdir()):
        raise ValueError(
            "La carpeta de salida contiene archivos; elija una carpeta nueva."
        )
    contenido = argumentos.csv.read_bytes()
    configuracion = ConfiguracionCSV(
        separador=argumentos.separador,
        decimal=argumentos.decimal,
        encoding=argumentos.encoding,
    )
    datos = CargadorCSV.cargar_bytes(contenido, configuracion)
    experimento = Regresion(
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
            f"Ningún modelo pudo evaluarse. Revise las configuraciones: {errores}"
        )
    archivos = construir_archivos_experimento(
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


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        salida = ejecutar(argumentos)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"No se pudo ejecutar LAB 03: {exc}\n")
    metadata = json.loads((salida / "configuracion.json").read_text(encoding="utf-8"))[
        "experimento"
    ]
    print(
        f"Resultados guardados en {salida}. RMSE de validación selecciona; prueba sólo evalúa."
    )
    if metadata["candidatos_fallidos"] or metadata["errores_prueba"]:
        print(
            f"Ejecución parcial: {metadata['candidatos_fallidos']} candidatos fallidos; "
            f"errores de prueba: {metadata['errores_prueba']}. Revise la evidencia exportada."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
