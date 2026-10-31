"""Herramientas de agrupamiento sobre :class:`framework_ia.modelos.no_supervisado.Cluster`.

Devuelven métricas y perfiles agregados; las etiquetas por fila nunca llegan al LLM.
"""

from __future__ import annotations

from ...modelos.no_supervisado import Cluster
from ...resultados import ResultadoCluster
from .registro import herramienta

MAX_K = 20
MIN_FILAS_PERFIL = 3  # el promedio de un grupo menor casi revela filas individuales
MAX_DISTINTIVAS = 3
ALGORITMOS = ("K-Means", "K-Medoids", "HAC")
METODOS_HAC = ("ward", "average", "complete", "single")


def crear_cluster(pizarra, **configuracion) -> Cluster:
    """Deja en la pizarra el modelo con la configuración activa del análisis."""
    cluster = Cluster(dataframe=pizarra.datos, **configuracion)
    pizarra.resultados["modelo_cluster"] = cluster
    return cluster


def _redondear(valor: float | None) -> float | None:
    return None if valor is None else round(float(valor), 4)


def _tabla(tabla) -> list[dict]:
    return tabla.round(4).astype(object).where(tabla.notna(), None).to_dict("records")


def resumir_resultado(resultado: ResultadoCluster) -> dict:
    """Tamaño de cada grupo y variables que más lo separan de la media, en desviaciones."""
    matriz = resultado.datos_preparados.matriz
    tamanos = resultado.etiquetas.value_counts().sort_index()
    desviaciones = matriz.std().replace(0, 1)
    medias = (matriz.groupby(resultado.etiquetas).mean() - matriz.mean()) / desviaciones
    grupos = []
    for grupo, tamano in tamanos.items():
        descripcion = {
            "cluster": int(grupo),
            "tamano": int(tamano),
            "proporcion": _redondear(tamano / len(matriz)),
        }
        if tamano >= MIN_FILAS_PERFIL:
            distintivas = medias.loc[grupo].sort_values(key=abs, ascending=False)
            descripcion["distintivas"] = {
                str(c): _redondear(v) for c, v in distintivas.head(MAX_DISTINTIVAS).items()
            }
        grupos.append(descripcion)
    return {
        "algoritmo": resultado.algoritmo,
        "n_clusters": int(len(tamanos)),
        "silhouette": _redondear(resultado.silhouette),
        "inercia": _redondear(resultado.inercia),
        "grupos": grupos,
    }


@herramienta(
    "Compara inercia y silhouette de K-Means para un intervalo de k.",
    k_min="Menor número de clusters, al menos 2.",
    k_max=f"Mayor número de clusters, hasta {MAX_K}.",
)
def evaluar_k(pizarra, k_min: int = 2, k_max: int = 10) -> dict:
    tabla = pizarra.resultados["modelo_cluster"].evaluar_kmeans(k_min, min(k_max, MAX_K))
    return {"filas": _tabla(tabla)}


@herramienta(
    "Compara silhouette de clustering jerárquico por método y número de clusters.",
    k_min="Menor número de clusters, al menos 2.",
    k_max=f"Mayor número de clusters, hasta {MAX_K}.",
    metodos=f"Subconjunto de {METODOS_HAC}; si se omite, todos.",
)
def evaluar_jerarquico(
    pizarra, k_min: int = 2, k_max: int = 10, metodos: list[str] | None = None
) -> dict:
    tabla = pizarra.resultados["modelo_cluster"].evaluar_hac(
        k_min, min(k_max, MAX_K), metodos=tuple(metodos or METODOS_HAC)
    )
    return {"filas": _tabla(tabla)}


@herramienta(
    "Agrupa con un algoritmo y describe los grupos con métricas y variables distintivas.",
    algoritmo=f"Uno de {ALGORITMOS}.",
    n_clusters="Número de clusters.",
    metodo_hac=f"Sólo para HAC: uno de {METODOS_HAC}; por defecto ward.",
)
def ejecutar_agrupamiento(
    pizarra, algoritmo: str, n_clusters: int, metodo_hac: str | None = None
) -> dict:
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo no soportado {algoritmo!r}; use {ALGORITMOS}.")
    opciones = {"metodo": metodo_hac or "ward"} if algoritmo == "HAC" else {}
    resultado = pizarra.resultados["modelo_cluster"].ajustar(
        algoritmo, n_clusters=n_clusters, **opciones
    )
    return resumir_resultado(resultado)


HERRAMIENTAS_AGRUPAMIENTO = (evaluar_k, evaluar_jerarquico, ejecutar_agrupamiento)
