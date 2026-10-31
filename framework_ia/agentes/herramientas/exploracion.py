"""Herramientas de análisis exploratorio sobre los métodos heredados de :class:`EDA`.

Todas devuelven resúmenes acotados (sin filas del dataset) para que el contexto
enviado al LLM sea pequeño y no exponga registros individuales.
"""

from __future__ import annotations

import pandas as pd

from ...utils import UmbralesEDA
from .registro import herramienta

MAX_FILAS = 20


def tipo_tarea(serie: pd.Series, umbrales: UmbralesEDA = UmbralesEDA()) -> str:
    """Clasificación si el target es categórico o numérico con pocos valores.

    ``Supervisado._tipo_target`` trata todo target numérico como regresión; aquí
    un 0/1 (p. ej. Potability) se reconoce como clasificación.
    """
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        if serie.nunique(dropna=True) > umbrales.max_clases_clasificacion:
            return "regresion"
    return "clasificacion"


def parece_identificador(serie: pd.Series, minimo_filas: int) -> bool:
    """Un valor distinto por fila, salvo decimales, que sí pueden ser continuos."""
    filas = len(serie)
    return (
        filas >= minimo_filas
        and serie.nunique(dropna=True) == filas
        and not pd.api.types.is_float_dtype(serie)
    )


def _redondear(valor: float) -> float:
    return round(float(valor), 4)


@herramienta(
    "Perfil del dataset: calidad, tipo de cada columna, nulos, cardinalidad y "
    "relación de las predictoras con el target."
)
def perfil_dataset(pizarra) -> dict:
    perfil: dict = {
        "calidad": pizarra.resumen_calidad(),
        "columnas": pizarra.resumen_columnas().reset_index().to_dict("records"),
    }
    if pizarra.target is not None:
        serie = pizarra.datos[pizarra.target]
        objetivo: dict = {"nombre": pizarra.target, "tarea": tipo_tarea(serie)}
        if objetivo["tarea"] == "clasificacion":
            resumen, _ = pizarra.balance_objetivo(pizarra.target)
            objetivo["clases"] = {
                str(clase): _redondear(proporcion)
                for clase, proporcion in resumen["Proporción"].items()
            }
        if pd.api.types.is_numeric_dtype(serie):
            numericas = pizarra.datos.select_dtypes(include="number").drop(
                columns=[pizarra.target]
            )
            correlaciones = numericas.corrwith(serie).dropna()
            objetivo["correlaciones"] = {
                str(c): _redondear(v)
                for c, v in correlaciones.sort_values(key=abs, ascending=False).items()
            }
        perfil["target"] = objetivo
    pizarra.resultados["perfil"] = perfil
    return perfil


@herramienta(
    "Pares de variables numéricas cuya correlación absoluta supera el umbral.",
    umbral="Correlación absoluta mínima, entre 0 y 1.",
    metodo="pearson, spearman o kendall.",
)
def correlaciones(pizarra, umbral: float = 0.8, metodo: str = "pearson") -> dict:
    if not 0 <= umbral <= 1:
        raise ValueError("umbral debe estar entre 0 y 1.")
    matriz = pizarra.datos.select_dtypes(include="number").corr(method=metodo)
    pares = []
    columnas = list(matriz.columns)
    for i, a in enumerate(columnas):
        for b in columnas[i + 1 :]:
            valor = matriz.loc[a, b]
            if pd.notna(valor) and abs(valor) >= umbral:
                pares.append({"a": str(a), "b": str(b), "r": _redondear(valor)})
    pares.sort(key=lambda par: abs(par["r"]), reverse=True)
    return {"metodo": metodo, "umbral": umbral, "pares": pares[:MAX_FILAS], "total": len(pares)}


@herramienta(
    "Valores atípicos por rango intercuartílico en columnas numéricas.",
    columnas="Columnas a revisar; si se omite, todas las numéricas.",
    factor_iqr="Multiplicador del IQR para los límites (1.5 es el estándar).",
)
def valores_atipicos(
    pizarra, columnas: list[str] | None = None, factor_iqr: float = 1.5
) -> dict:
    resumen = pizarra.detectar_outliers(columnas, factor_iqr)["resumen"]
    resumen = resumen.sort_values("porcentaje_outliers", ascending=False).head(MAX_FILAS)
    return {
        "factor_iqr": factor_iqr,
        "columnas": [
            {
                "columna": str(columna),
                "cantidad": int(fila["cantidad_outliers"]),
                "porcentaje": float(fila["porcentaje_outliers"]),
                "limite_inferior": _redondear(fila["limite_inferior"]),
                "limite_superior": _redondear(fila["limite_superior"]),
            }
            for columna, fila in resumen.iterrows()
        ],
    }


@herramienta(
    "Frecuencias de los valores más comunes de una columna.",
    columna="Nombre exacto de la columna.",
    top="Cantidad máxima de valores a devolver.",
)
def frecuencias_columna(pizarra, columna: str, top: int = 10) -> dict:
    if columna not in pizarra.datos.columns:
        raise KeyError(f"No existe la columna {columna!r}.")
    tabla = pizarra.frecuencias([columna])[columna].head(top)
    return {
        "columna": columna,
        "valores_unicos": int(pizarra.datos[columna].nunique(dropna=False)),
        "frecuencias": [
            {"valor": str(valor), "frecuencia": int(f["frecuencia"]), "porcentaje": float(f["porcentaje"])}
            for valor, f in tabla.iterrows()
        ],
    }


HERRAMIENTAS_EXPLORACION = (perfil_dataset, correlaciones, valores_atipicos, frecuencias_columna)
