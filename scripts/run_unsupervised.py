"""Run reproducible unsupervised learning experiments (PCA, HAC, K-Means, t-SNE, UMAP)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

from ai_framework.data.eda import EDA
from ai_framework.models.unsupervised import Cluster, DependenciaOpcionalError
from ai_framework.utils.unsupervised import evaluar_hac, evaluar_kmeans

__all__ = ["run", "ejecutar"]


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible results for PCA, HAC, K-Means, t-SNE, and UMAP."
    )
    parser.add_argument("csv", type=Path, help="CSV file path to analyze.")
    parser.add_argument("--salida", "--output", dest="salida", type=Path, default=Path("salida_no_supervisado"))
    parser.add_argument("--separador", "--separator", dest="separador", default=",", help="CSV delimiter.")
    parser.add_argument("--decimal", default=".", help="CSV decimal separator.")
    parser.add_argument(
        "--features",
        nargs="+",
        help="Feature columns. Defaults to all numeric columns.",
    )
    parser.add_argument("--semilla", "--seed", dest="semilla", type=int, default=42)
    parser.add_argument("--sin-escalar", "--no-scaling", dest="sin_escalar", action="store_true")
    parser.add_argument("--incluir-categoricas", "--include-categorical", dest="incluir_categoricas", action="store_true")
    return parser.parse_args()


def _json(valor: Any) -> Any:
    """Convert NumPy/Pandas types to JSON-serializable values."""
    if isinstance(valor, (np.integer, np.floating)):
        return valor.item()
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, (pd.Index, np.ndarray)):
        return valor.tolist()
    raise TypeError(f"Non-serializable type: {type(valor).__name__}")


def _confiabilidad(matriz: pd.DataFrame, coordenadas: pd.DataFrame) -> float | None:
    """Compute trustworthiness metric as visual analysis support."""
    vecinos = min(5, (len(matriz) - 1) // 2)
    if vecinos < 1:
        return None
    return float(
        trustworthiness(matriz.to_numpy(), coordenadas.to_numpy(), n_neighbors=vecinos)
    )


def _crear_reductor(datos: pd.DataFrame, configuracion: dict[str, Any]) -> Cluster:
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def _crear_cluster(datos: pd.DataFrame, configuracion: dict[str, Any]) -> Cluster:
    return Cluster(
        dataframe=datos,
        features=configuracion["features"],
        incluir_categoricas=configuracion["incluir_categoricas"],
        estandarizar=configuracion["estandarizar"],
    )


def _guardar_json(ruta: Path, contenido: dict[str, Any]) -> None:
    ruta.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, default=_json) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    args = _argumentos()
    datos = pd.read_csv(args.csv, sep=args.separador, decimal=args.decimal)
    eda = EDA(dataframe=datos)
    features = (
        args.features
        or eda.seleccionar_variables().select_dtypes(include="number").columns.tolist()
    )
    if not features:
        features = datos.select_dtypes(include="number").columns.tolist()
    if not features:
        raise ValueError("CSV contains no numeric columns; specify --features.")

    filas_totales, columnas_totales = eda.dimensiones()
    configuracion = {
        "archivo": args.csv.name,
        "filas": int(filas_totales),
        "columnas": int(columnas_totales),
        "features": features,
        "incluir_categoricas": args.incluir_categoricas,
        "estandarizar": not args.sin_escalar,
        "semilla": args.semilla,
    }
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    eda.guardar_csv(str(salida / "dataset_analizado.csv"))
    _guardar_json(salida / "configuracion.json", configuracion)
    resumen: dict[str, Any] = {
        "configuracion": configuracion,
        "calidad": eda.resumen_calidad(),
        "primeras_filas": len(eda.primeras_filas(5)),
        "ultimas_filas": len(eda.ultimas_filas(5)),
    }

    reductor = _crear_reductor(datos, configuracion)
    componentes_maximos = min(5, len(datos), len(features))
    acp_filas = []
    for componentes in range(2, componentes_maximos + 1):
        resultado = reductor.ACP(componentes)
        resultado.coordenadas.to_csv(
            salida / f"acp_{componentes}_componentes.csv", index=True
        )
        acp_filas.append(
            {
                "componentes": componentes,
                "varianza_acumulada_pct": float(
                    resultado.varianza_acumulada.iloc[-1]
                ),
            }
        )
    tabla_acp = pd.DataFrame(acp_filas)
    tabla_acp.to_csv(salida / "comparacion_acp.csv", index=False)
    resumen["acp"] = tabla_acp.to_dict(orient="records")

    cluster = _crear_cluster(datos, configuracion)
    matriz_cluster = cluster.preparar_matriz().matriz
    k_max = min(10, len(datos) - 1)
    tabla_kmeans = evaluar_kmeans(matriz_cluster, 2, k_max, random_state=args.semilla)
    tabla_kmeans.to_csv(salida / "comparacion_kmeans.csv", index=False)
    mejor_kmeans = tabla_kmeans.loc[tabla_kmeans["silhouette"].idxmax()]
    resultado_kmeans = cluster.K_means(
        int(mejor_kmeans["k"]), random_state=args.semilla
    )
    resultado_kmeans.etiquetas.to_csv(salida / "kmeans_asignaciones.csv", header=True)
    resumen["kmeans"] = {
        "comparacion": tabla_kmeans.to_dict(orient="records"),
        "seleccion_sugerida": mejor_kmeans.to_dict(),
    }

    tabla_hac = evaluar_hac(matriz_cluster, 2, k_max)
    tabla_hac.to_csv(salida / "comparacion_hac.csv", index=False)
    mejor_hac = tabla_hac.loc[tabla_hac["silhouette"].idxmax()]
    resultado_hac = cluster.HAC(int(mejor_hac["k"]), metodo=str(mejor_hac["metodo"]))
    resultado_hac.etiquetas.to_csv(salida / "hac_asignaciones.csv", header=True)
    resumen["hac"] = {
        "comparacion": tabla_hac.to_dict(orient="records"),
        "seleccion_sugerida": mejor_hac.to_dict(),
    }

    tabla_benchmark = cluster.benchmark()
    tabla_benchmark.to_csv(salida / "comparacion_algoritmos_cluster.csv", index=False)
    resumen["benchmark_cluster"] = tabla_benchmark.to_dict(orient="records")

    matriz = reductor.preparar_matriz().matriz
    filas_tsne = []
    for perplexity in (5, 15, 30):
        if perplexity >= len(datos):
            continue
        resultado = reductor.TSNE(
            perplexity=perplexity, max_iter=1000, random_state=args.semilla
        )
        resultado.coordenadas.to_csv(
            salida / f"tsne_perplexity_{perplexity}.csv", index=True
        )
        filas_tsne.append(
            {
                "perplexity": perplexity,
                "trustworthiness": _confiabilidad(matriz, resultado.coordenadas),
            }
        )
    tabla_tsne = pd.DataFrame(filas_tsne)
    tabla_tsne.to_csv(salida / "comparacion_tsne.csv", index=False)
    resumen["tsne"] = tabla_tsne.to_dict(orient="records")

    filas_umap = []
    try:
        for vecinos in (5, 10, 15):
            if vecinos >= len(datos):
                continue
            resultado = reductor.UMAP(
                n_neighbors=vecinos, min_dist=0.1, random_state=args.semilla
            )
            resultado.coordenadas.to_csv(
                salida / f"umap_vecinos_{vecinos}.csv", index=True
            )
            filas_umap.append(
                {
                    "n_neighbors": vecinos,
                    "min_dist": 0.1,
                    "trustworthiness": _confiabilidad(matriz, resultado.coordenadas),
                }
            )
        tabla_umap = pd.DataFrame(filas_umap)
        tabla_umap.to_csv(salida / "comparacion_umap.csv", index=False)
        resumen["umap"] = tabla_umap.to_dict(orient="records")
    except DependenciaOpcionalError as exc:
        resumen["umap"] = {"error": str(exc)}

    _guardar_json(salida / "resumen_experimentos.json", resumen)
    print(f"Experiments finished. Results: {salida.resolve()}")


ejecutar = run

if __name__ == "__main__":
    run()
