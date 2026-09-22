"""Ejecuta un conjunto reproducible de experimentos para el Lab01.

El script no sustituye el análisis escrito: genera las tablas, proyecciones y
configuraciones que deben interpretarse y citarse en el informe.  No hay rutas
ni columnas codificadas; por omisión se usan las columnas numéricas del CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

from framework_ia.modelos.cluster import Cluster
from framework_ia.modelos.reduccion_dimensional import (
    DependenciaOpcionalError,
    ReduccionDimensional,
)


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera resultados reproducibles de PCA, HAC, K-Means, t-SNE y UMAP."
    )
    parser.add_argument("csv", type=Path, help="Archivo CSV que se analizará.")
    parser.add_argument("--salida", type=Path, default=Path("salida_lab01"))
    parser.add_argument("--separador", default=",", help="Separador del CSV.")
    parser.add_argument("--decimal", default=".", help="Separador decimal del CSV.")
    parser.add_argument(
        "--features",
        nargs="+",
        help="Columnas a analizar. Si se omite, se usan todas las numéricas.",
    )
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-escalar", action="store_true")
    parser.add_argument("--incluir-categoricas", action="store_true")
    return parser.parse_args()


def _json(valor: Any) -> Any:
    """Convierte tipos de NumPy/Pandas a valores que JSON puede persistir."""
    if isinstance(valor, (np.integer, np.floating)):
        return valor.item()
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, (pd.Index, np.ndarray)):
        return valor.tolist()
    raise TypeError(f"Tipo no serializable: {type(valor).__name__}")


def _confiabilidad(matriz: pd.DataFrame, coordenadas: pd.DataFrame) -> float | None:
    """Calcula trustworthiness como apoyo, no sustituto del análisis visual."""
    vecinos = min(5, (len(matriz) - 1) // 2)
    if vecinos < 1:
        return None
    return float(
        trustworthiness(matriz.to_numpy(), coordenadas.to_numpy(), n_neighbors=vecinos)
    )


def _crear_reductor(datos: pd.DataFrame, configuracion: dict[str, Any]) -> ReduccionDimensional:
    return ReduccionDimensional(
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
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False, default=_json) + "\n", encoding="utf-8")


def ejecutar() -> None:
    args = _argumentos()
    datos = pd.read_csv(args.csv, sep=args.separador, decimal=args.decimal)
    features = args.features or datos.select_dtypes(include="number").columns.tolist()
    if not features:
        raise ValueError("El CSV no contiene columnas numéricas; indique --features.")

    configuracion = {
        "archivo": args.csv.name,
        "filas": int(len(datos)),
        "features": features,
        "incluir_categoricas": args.incluir_categoricas,
        "estandarizar": not args.sin_escalar,
        "semilla": args.semilla,
    }
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    _guardar_json(salida / "configuracion.json", configuracion)
    resumen: dict[str, Any] = {"configuracion": configuracion}

    reductor = _crear_reductor(datos, configuracion)
    componentes_maximos = min(5, len(datos), len(features))
    acp_filas = []
    for componentes in range(2, componentes_maximos + 1):
        resultado = reductor.ACP(componentes)
        resultado.coordenadas.to_csv(salida / f"acp_{componentes}_componentes.csv", index=True)
        acp_filas.append(
            {
                "componentes": componentes,
                "varianza_acumulada_pct": float(resultado.varianza_acumulada.iloc[-1]),
            }
        )
    tabla_acp = pd.DataFrame(acp_filas)
    tabla_acp.to_csv(salida / "comparacion_acp.csv", index=False)
    resumen["acp"] = tabla_acp.to_dict(orient="records")

    cluster = _crear_cluster(datos, configuracion)
    k_max = min(10, len(datos) - 1)
    tabla_kmeans = cluster.evaluar_kmeans(2, k_max, random_state=args.semilla)
    tabla_kmeans.to_csv(salida / "comparacion_kmeans.csv", index=False)
    mejor_kmeans = tabla_kmeans.loc[tabla_kmeans["silhouette"].idxmax()]
    resultado_kmeans = cluster.K_means(int(mejor_kmeans["k"]), random_state=args.semilla)
    resultado_kmeans.etiquetas.to_csv(salida / "kmeans_asignaciones.csv", header=True)
    resumen["kmeans"] = {
        "comparacion": tabla_kmeans.to_dict(orient="records"),
        "seleccion_sugerida": mejor_kmeans.to_dict(),
    }

    tabla_hac = cluster.evaluar_hac(2, k_max)
    tabla_hac.to_csv(salida / "comparacion_hac.csv", index=False)
    mejor_hac = tabla_hac.loc[tabla_hac["silhouette"].idxmax()]
    resultado_hac = cluster.HAC(int(mejor_hac["k"]), metodo=str(mejor_hac["metodo"]))
    resultado_hac.etiquetas.to_csv(salida / "hac_asignaciones.csv", header=True)
    resumen["hac"] = {
        "comparacion": tabla_hac.to_dict(orient="records"),
        "seleccion_sugerida": mejor_hac.to_dict(),
    }

    matriz = reductor.preparar_matriz().matriz
    filas_tsne = []
    for perplexity in (5, 15, 30):
        if perplexity >= len(datos):
            continue
        resultado = reductor.TSNE(
            perplexity=perplexity, max_iter=1000, random_state=args.semilla
        )
        resultado.coordenadas.to_csv(salida / f"tsne_perplexity_{perplexity}.csv", index=True)
        filas_tsne.append(
            {"perplexity": perplexity, "trustworthiness": _confiabilidad(matriz, resultado.coordenadas)}
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
            resultado.coordenadas.to_csv(salida / f"umap_vecinos_{vecinos}.csv", index=True)
            filas_umap.append(
                {"n_neighbors": vecinos, "min_dist": 0.1, "trustworthiness": _confiabilidad(matriz, resultado.coordenadas)}
            )
        tabla_umap = pd.DataFrame(filas_umap)
        tabla_umap.to_csv(salida / "comparacion_umap.csv", index=False)
        resumen["umap"] = tabla_umap.to_dict(orient="records")
    except DependenciaOpcionalError as exc:
        resumen["umap"] = {"error": str(exc)}

    _guardar_json(salida / "resumen_experimentos.json", resumen)
    print(f"Experimentos completados. Resultados: {salida.resolve()}")


if __name__ == "__main__":
    ejecutar()
