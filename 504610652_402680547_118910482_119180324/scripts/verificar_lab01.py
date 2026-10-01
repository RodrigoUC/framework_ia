"""Comprueba integridad y completitud de los resultados generados de Lab01."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def verificar(resultados: Path) -> None:
    resumen = json.loads((resultados / "resumen_experimentos.json").read_text(encoding="utf-8"))
    configuracion = json.loads((resultados / "configuracion.json").read_text(encoding="utf-8"))
    if resumen.get("estado") != "completado" or resumen.get("error"):
        raise ValueError("La ejecución no terminó correctamente.")
    if resumen["configuracion"] != configuracion:
        raise ValueError("La configuración no coincide con el resumen.")
    for metodo in ("acp", "kmeans", "hac", "tsne", "umap"):
        if not resumen.get(metodo) or (isinstance(resumen[metodo], dict) and resumen[metodo].get("error")):
            raise ValueError(f"Falta un resultado válido de {metodo}.")
    archivos = resumen.get("archivos", {})
    requeridos = {"matriz_preparada.csv", "eda_descriptivos.csv", "eda_nulos.csv", "eda_correlacion.csv",
        "comparacion_acp.csv", "comparacion_kmeans.csv", "comparacion_hac.csv", "comparacion_tsne.csv",
        "comparacion_umap.csv", "kmeans_asignaciones.csv", "hac_asignaciones.csv", "hac_linkage.csv"}
    if not requeridos <= archivos.keys():
        raise ValueError(f"Faltan archivos obligatorios: {sorted(requeridos - archivos.keys())}")
    tablas = {}
    for nombre, metadata in archivos.items():
        ruta = resultados / nombre
        if Path(nombre).name != nombre or not ruta.is_file():
            raise ValueError(f"Archivo ausente o ruta inválida: {nombre}")
        if hashlib.sha256(ruta.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Archivo modificado desde la ejecución: {nombre}")
        with ruta.open(encoding="utf-8", newline="") as archivo:
            tabla = list(csv.DictReader(archivo))
        if len(tabla) != metadata["filas"]:
            raise ValueError(f"Cantidad de filas inconsistente: {nombre}")
        tablas[nombre] = tabla
    filas = configuracion["filas"]
    for metodo in ("acp", "kmeans", "hac", "tsne", "umap"):
        esperadas = resumen[metodo]["comparacion"] if metodo in ("kmeans", "hac") else resumen[metodo]
        tabla = tablas[f"comparacion_{metodo}.csv"]
        if len(tabla) != len(esperadas):
            raise ValueError(f"Comparación CSV/JSON inconsistente de {metodo}.")
        for real, esperada in zip(tabla, esperadas):
            for columna, valor in esperada.items():
                if columna not in real:
                    raise ValueError(f"Falta columna {columna} de {metodo}.")
                celda = real[columna]
                coincide = (celda == "" if valor is None else
                    math.isclose(float(celda), valor, rel_tol=1e-10, abs_tol=1e-12) if isinstance(valor, (int, float)) else celda == str(valor))
                if not coincide:
                    raise ValueError(f"Valor CSV/JSON inconsistente de {metodo}: {columna}.")
    if {int(r["k"]) for r in resumen["kmeans"]["comparacion"]} != set(range(2, min(10, filas - 1) + 1)):
        raise ValueError("El intervalo K-Means está incompleto.")
    if {(r["metodo"], int(r["k"])) for r in resumen["hac"]["comparacion"]} != {(m, k) for m in ("ward", "average", "complete", "single") for k in range(2, min(10, filas - 1) + 1)}:
        raise ValueError("El intervalo HAC está incompleto.")
    if {r["perplexity"] for r in resumen["tsne"]} != {p for p in (5, 15, 30) if p < filas}:
        raise ValueError("El intervalo t-SNE está incompleto.")
    if {(r["n_neighbors"], r["min_dist"]) for r in resumen["umap"]} != {(n, d) for n in (5, 10, 15) if n < filas for d in (0.0, 0.1, 0.5)}:
        raise ValueError("El intervalo UMAP está incompleto.")
    if len(tablas["matriz_preparada.csv"]) != filas:
        raise ValueError("La matriz no contiene todas las observaciones.")
    for metodo in ("kmeans", "hac"):
        asignaciones = tablas[f"{metodo}_asignaciones.csv"]
        if len(asignaciones) != filas or any(not r.get("cluster") for r in asignaciones):
            raise ValueError(f"Asignaciones incompletas de {metodo}.")
        comparacion = resumen[metodo]["comparacion"]
        seleccion = resumen[metodo]["seleccion_sugerida"]
        if seleccion not in comparacion or not math.isfinite(seleccion["silhouette"]):
            raise ValueError(f"Selección inválida de {metodo}.")
        for nombre in (f"{metodo}_centroides_preparados.csv", f"{metodo}_perfiles_originales.csv"):
            if nombre not in archivos:
                raise ValueError(f"Falta {nombre}.")
        cantidad_grupos = len({r["cluster"] for r in asignaciones})
        if len(tablas[f"{metodo}_centroides_preparados.csv"]) != cantidad_grupos or sum(int(r["cantidad"]) for r in tablas[f"{metodo}_perfiles_originales.csv"]) != filas:
            raise ValueError(f"Perfiles o centroides inconsistentes de {metodo}.")
    if len(tablas["hac_linkage.csv"]) != filas - 1:
        raise ValueError("El árbol HAC no contiene n-1 fusiones.")
    for fila in resumen["acp"]:
        componentes = int(fila["componentes"])
        for nombre in (f"acp_{componentes}_componentes.csv", f"acp_{componentes}_cargas.csv", f"acp_{componentes}_varianza.csv"):
            if nombre not in archivos:
                raise ValueError(f"Falta {nombre}.")
        if len(tablas[f"acp_{componentes}_componentes.csv"]) != filas:
            raise ValueError("Coordenadas ACP incompletas.")
    if resumen.get("selecciones", {}).get("acp") not in resumen["acp"]:
        raise ValueError("Selección ACP inválida.")
    for metodo in ("tsne", "umap"):
        comparacion = resumen[metodo]
        if resumen.get("selecciones", {}).get(metodo) not in comparacion:
            raise ValueError(f"Selección inválida de {metodo}.")
        for fila in comparacion:
            if not 0 <= fila["trustworthiness"] <= 1:
                raise ValueError(f"Trustworthiness inválido de {metodo}.")
            if fila["archivo"] not in tablas or len(tablas[fila["archivo"]]) != filas:
                raise ValueError(f"Coordenadas incompletas de {metodo}.")
            for coordenada in tablas[fila["archivo"]]:
                if any(not math.isfinite(float(v)) for k, v in coordenada.items() if k):
                    raise ValueError(f"Coordenadas no finitas de {metodo}.")
    print(f"Verificación correcta: cinco métodos completos, {len(archivos)} CSV íntegros y {filas} observaciones.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultados", type=Path, default=Path("resultados"))
    args = parser.parse_args()
    try:
        verificar(args.resultados)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Verificación fallida: {exc}\n")


if __name__ == "__main__":
    main()
