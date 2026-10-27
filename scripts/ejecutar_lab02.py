"""Ejecuta clasificación reproducible sobre un CSV, sin acoplarse a Streamlit.

Las salidas son evidencia experimental, no un informe ni una recomendación
médica o de seguridad del agua. Nunca se descarga ni publica el dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from framework_ia.datos.fuentes import CargadorCSV
from framework_ia.exportacion import _json_seguro, construir_archivos_experimento
from framework_ia.modelos.supervisado import Clasificacion
from framework_ia.utils import ConfiguracionCSV

__all__ = ["_json_seguro", "crear_parser", "ejecutar", "main"]


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
    parser.add_argument(
        "--estrategia",
        choices=["cv", "holdout"],
        default="cv",
        help="cv: validación cruzada estratificada en train (predeterminada); holdout: una validación.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Pliegues de la validación cruzada.")
    parser.add_argument(
        "--seleccion-k",
        default="auto",
        help="Atributos a conservar en la etapa de selección; 'auto' usa la mitad, 0 la omite.",
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
        },
    )
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    for nombre, contenido_archivo in archivos.items():
        destino = argumentos.salida / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido_archivo)
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
