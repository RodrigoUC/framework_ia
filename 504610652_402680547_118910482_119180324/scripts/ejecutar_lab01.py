"""Genera evidencia reproducible para Lab01; no sustituye el análisis escrito."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.manifold import trustworthiness

from framework_ia.modelos.no_supervisado import Cluster


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimentos de ACP, HAC, K-Means, t-SNE y UMAP.")
    parser.add_argument("csv", type=Path, help="Archivo CSV que se analizará.")
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    parser.add_argument("--separador", default=",")
    parser.add_argument("--decimal", default=".")
    parser.add_argument("--features", nargs="+", help="Columnas; por omisión, todas las numéricas.")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-escalar", action="store_true")
    parser.add_argument("--incluir-categoricas", action="store_true")
    return parser.parse_args()


def _normalizar(valor: Any) -> Any:
    """Evita NaN/Infinity no estándar en JSON y convierte tipos NumPy."""
    if isinstance(valor, dict):
        return {str(k): _normalizar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, pd.Index, np.ndarray)):
        return [_normalizar(v) for v in valor]
    if isinstance(valor, (np.integer, np.floating)):
        return _normalizar(valor.item())
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    if isinstance(valor, Path):
        return str(valor)
    return valor


def _guardar_json(ruta: Path, contenido: dict[str, Any]) -> None:
    ruta.write_text(json.dumps(_normalizar(contenido), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _confiabilidad(matriz: pd.DataFrame, coordenadas: pd.DataFrame) -> float:
    vecinos = min(5, (len(matriz) - 1) // 2)
    return float(trustworthiness(matriz.to_numpy(), coordenadas.to_numpy(), n_neighbors=vecinos))


def _mejor(tabla: pd.DataFrame, metrica: str, desempate: list[str]) -> dict[str, Any]:
    validas = tabla.loc[np.isfinite(tabla[metrica])]
    if validas.empty:
        raise ValueError(f"No hay resultados válidos para seleccionar por {metrica}.")
    return validas.sort_values([metrica, *desempate], ascending=[False, *([True] * len(desempate))], kind="stable").iloc[0].to_dict()


def _versiones() -> dict[str, str]:
    versiones = {}
    for paquete in ("numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "numba"):
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            versiones[paquete] = "no instalado"
    return versiones


def ejecutar() -> None:
    args = _argumentos()
    # Validar antes de crear archivos: una entrada inválida no deja resultados nuevos.
    if not args.csv.is_file():
        raise FileNotFoundError(f"No existe el CSV: {args.csv}")
    datos = pd.read_csv(args.csv, sep=args.separador, decimal=args.decimal)
    features = args.features or (datos.columns.tolist() if args.incluir_categoricas else datos.select_dtypes(include="number").columns.tolist())
    if not features:
        raise ValueError("Seleccione columnas numéricas o active --incluir-categoricas.")
    if len(datos) < 6:
        raise ValueError("Este conjunto de experimentos requiere al menos seis filas (perplexity y vecinos 5).")
    cluster = Cluster(dataframe=datos, features=features, incluir_categoricas=args.incluir_categoricas, estandarizar=not args.sin_escalar)
    preparados = cluster.preparar_matriz()
    matriz = preparados.matriz
    if matriz.shape[1] < 2:
        raise ValueError("Se requieren al menos dos variables preparadas para comparar ACP.")
    if len(matriz.drop_duplicates()) < min(10, len(datos) - 1):
        raise ValueError("Hay menos observaciones distintas que grupos del intervalo K-Means; revise duplicados o datos.")
    if args.salida.resolve() == args.csv.resolve().parent:
        raise ValueError("Use una carpeta de resultados distinta de la carpeta del CSV.")

    configuracion = {
        "archivo": args.csv.name, "sha256_datos": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "filas": len(datos), "columnas": datos.columns.tolist(), "features": features,
        "separador": args.separador, "decimal": args.decimal,
        "incluir_categoricas": args.incluir_categoricas, "estandarizar": not args.sin_escalar,
        "semilla": args.semilla, "python": platform.python_version(), "versiones": _versiones(),
        "preprocesamiento": {
            "numericas": "imputación por mediana; StandardScaler si estandarizar=true",
            "categoricas": "imputación por moda y one-hot si incluir_categoricas=true",
            "infinitos": "tratados como faltantes", "constantes": "descartadas",
            "columnas_origen": list(preparados.columnas_origen),
            "columnas_descartadas": list(preparados.columnas_descartadas),
            "columnas_preparadas": matriz.columns.tolist(),
            "categoricas_excluidas": [c for c in datos.select_dtypes(exclude="number").columns if c not in features],
        },
    }
    bases = {"acp": {"componentes": 2}, "kmeans": {"k": 3, "n_init": 20, "max_iter": 500},
             "hac": {"metodo": "ward", "k": 3, "metrica": "euclidean"},
             "tsne": {"perplexity": 30, "max_iter": 1000, "base_operativa": 5},
             "umap": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2}}
    resumen: dict[str, Any] = {"estado": "en_proceso", "configuracion": configuracion,
        "configuraciones_base": bases, "omisiones": [], "selecciones": {},
        "criterios": {"acp": "mínimo número de componentes que alcanza 95% de varianza; si no, máximo evaluado",
            "kmeans": "máximo silhouette; empate: menor k",
            "hac": "máximo silhouette; empate: menor k y nombre de método en orden alfabético",
            "tsne": "máximo trustworthiness; empate: menor perplexity",
            "umap": "máximo trustworthiness; empate: menor n_neighbors y menor min_dist",
            "advertencia": "Trustworthiness evalúa preservación local, no demuestra grupos reales ni conservación global.",
            "vecinos_trustworthiness": min(5, (len(datos) - 1) // 2)}, "archivos": {}}
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)

    def csv(nombre: str, tabla: pd.DataFrame | pd.Series, *, index: bool = True) -> None:
        ruta = salida / nombre
        tabla.to_csv(ruta, index=index, encoding="utf-8")
        resumen["archivos"][nombre] = {"filas": len(tabla), "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()}

    def guardar_grupos(nombre: str, resultado: Any) -> None:
        csv(f"{nombre}_asignaciones.csv", resultado.etiquetas)
        csv(f"{nombre}_centroides_preparados.csv", resultado.centroides)
        numericas = datos.select_dtypes(include="number")
        perfiles = numericas.groupby(resultado.etiquetas, sort=True).mean()
        perfiles.insert(0, "cantidad", resultado.etiquetas.value_counts().sort_index())
        csv(f"{nombre}_perfiles_originales.csv", perfiles)

    _guardar_json(salida / "configuracion.json", configuracion)
    # Sobrescribe inmediatamente el resumen: un fallo no conserva un éxito anterior.
    _guardar_json(salida / "resumen_experimentos.json", resumen)
    try:
        csv("matriz_preparada.csv", matriz)
        csv("eda_descriptivos.csv", datos.describe(include="all").T)
        csv("eda_nulos.csv", pd.DataFrame({"nulos": datos.isna().sum(), "porcentaje": datos.isna().mean() * 100}))
        csv("eda_correlacion.csv", datos.select_dtypes(include="number").corr())
        csv("eda_frecuencias_categoricas.csv", pd.DataFrame([
            {"columna": c, "categoria": str(v), "cantidad": int(n)}
            for c in datos.select_dtypes(exclude="number") for v, n in datos[c].value_counts(dropna=False).items()
        ], columns=["columna", "categoria", "cantidad"]), index=False)

        filas_acp = []
        for componentes in range(2, min(4, *matriz.shape) + 1):
            resultado = cluster.ACP(componentes)
            csv(f"acp_{componentes}_componentes.csv", resultado.coordenadas)
            csv(f"acp_{componentes}_cargas.csv", resultado.cargas)
            csv(f"acp_{componentes}_varianza.csv", pd.concat([resultado.varianza_explicada, resultado.varianza_acumulada], axis=1))
            filas_acp.append({"componentes": componentes, "varianza_acumulada_pct": float(resultado.varianza_acumulada.iloc[-1])})
        tabla_acp = pd.DataFrame(filas_acp)
        csv("comparacion_acp.csv", tabla_acp, index=False)
        resumen["acp"] = tabla_acp.to_dict(orient="records")
        candidatas = tabla_acp.loc[tabla_acp.varianza_acumulada_pct >= 95]
        resumen["selecciones"]["acp"] = (candidatas.iloc[0] if not candidatas.empty else tabla_acp.iloc[-1]).to_dict()

        k_max = min(10, len(datos) - 1)
        tabla_kmeans = cluster.evaluar_kmeans(2, k_max, random_state=args.semilla)
        csv("comparacion_kmeans.csv", tabla_kmeans, index=False)
        mejor_kmeans = _mejor(tabla_kmeans, "silhouette", ["k"])
        guardar_grupos("kmeans", cluster.K_means(int(mejor_kmeans["k"]), random_state=args.semilla))
        resumen["kmeans"] = {"comparacion": tabla_kmeans.to_dict(orient="records"), "seleccion_sugerida": mejor_kmeans}
        tabla_hac = cluster.evaluar_hac(2, k_max)
        arboles_hac = {m: linkage(matriz.to_numpy(), method=m, metric="euclidean") for m in ("ward", "average", "complete", "single")}
        tabla_hac["k_real"] = [len(np.unique(fcluster(arboles_hac[fila.metodo], t=int(fila.k), criterion="maxclust"))) for fila in tabla_hac.itertuples()]
        csv("comparacion_hac.csv", tabla_hac, index=False)
        mejor_hac = _mejor(tabla_hac, "silhouette", ["k", "metodo"])
        resultado_hac = cluster.HAC(int(mejor_hac["k"]), metodo=str(mejor_hac["metodo"]))
        guardar_grupos("hac", resultado_hac)
        columnas_linkage = ["nodo_izquierdo", "nodo_derecho", "distancia", "cantidad"]
        csv("hac_linkage.csv", pd.DataFrame(resultado_hac.matriz_vinculacion, columns=columnas_linkage), index=False)
        for metodo in ("ward", "average", "complete", "single"):
            csv(f"hac_linkage_{metodo}.csv", pd.DataFrame(arboles_hac[metodo], columns=columnas_linkage), index=False)
        resumen["hac"] = {"comparacion": tabla_hac.to_dict(orient="records"), "seleccion_sugerida": mejor_hac,
                          "grupos_obtenidos": int(resultado_hac.etiquetas.nunique())}
        bases["hac"]["k_real"] = int(tabla_hac.loc[(tabla_hac.metodo == "ward") & (tabla_hac.k == 3), "k_real"].iloc[0])

        filas_tsne = []
        for perplexity in (5, 15, 30):
            if perplexity >= len(datos):
                resumen["omisiones"].append({"metodo": "tsne", "perplexity": perplexity, "motivo": f"Debe ser menor que las {len(datos)} filas."})
                continue
            resultado = cluster.TSNE(perplexity=perplexity, max_iter=1000, random_state=args.semilla)
            archivo = f"tsne_perplexity_{perplexity}.csv"
            csv(archivo, resultado.coordenadas)
            filas_tsne.append({"perplexity": perplexity, "trustworthiness": _confiabilidad(matriz, resultado.coordenadas), "archivo": archivo})
        tabla_tsne = pd.DataFrame(filas_tsne)
        csv("comparacion_tsne.csv", tabla_tsne, index=False)
        resumen["tsne"] = tabla_tsne.to_dict(orient="records")
        resumen["selecciones"]["tsne"] = _mejor(tabla_tsne, "trustworthiness", ["perplexity"])

        filas_umap = []
        for vecinos in (5, 10, 15):
            if vecinos >= len(datos):
                resumen["omisiones"].append({"metodo": "umap", "n_neighbors": vecinos, "motivo": f"Debe ser menor que las {len(datos)} filas."})
                continue
            for min_dist in (0.0, 0.1, 0.5):
                resultado = cluster.UMAP(n_neighbors=vecinos, min_dist=min_dist, random_state=args.semilla)
                archivo = f"umap_vecinos_{vecinos}_min_dist_{str(min_dist).replace('.', '_')}.csv"
                csv(archivo, resultado.coordenadas)
                filas_umap.append({"n_neighbors": vecinos, "min_dist": min_dist, "trustworthiness": _confiabilidad(matriz, resultado.coordenadas), "archivo": archivo})
        tabla_umap = pd.DataFrame(filas_umap)
        csv("comparacion_umap.csv", tabla_umap, index=False)
        resumen["umap"] = tabla_umap.to_dict(orient="records")
        resumen["selecciones"]["umap"] = _mejor(tabla_umap, "trustworthiness", ["n_neighbors", "min_dist"])
        resumen["estado"] = "completado"
    except Exception as exc:
        resumen["estado"] = "error"
        resumen["error"] = f"{type(exc).__name__}: {exc}"
        _guardar_json(salida / "resumen_experimentos.json", resumen)
        raise
    _guardar_json(salida / "resumen_experimentos.json", resumen)
    print(f"Experimentos completados. Resultados: {salida.resolve()}")


if __name__ == "__main__":
    ejecutar()
