"""Catálogo de rutas, vistas, paletas y opciones de configuración de la interfaz."""

from __future__ import annotations

from pathlib import Path

from framework_ia.utils import configuraciones_experimento
from framework_ia.utils.supervisado.configuracion_regresion import ALGORITMOS_LAB3

BASE_DIR = Path(__file__).resolve().parents[3]
APP_NAME = "Nexo IA"

VISTAS_REGRESORES = {f"regresion_{alg.lower()}": alg for alg in ALGORITMOS_LAB3}
VISTAS_REGRESION = {
    *VISTAS_REGRESORES,
    "regresion",
    "regresion_configuracion",
    "lab3_benchmark",
    "lab3_comparacion",
    "lab3_resultados",
}

PALETA = {
    "fondo": "#f7f7f5",
    "superficie": "#eeeee9",
    "texto": "#242523",
    "acento": "#1a3c2b",
    "acento_alt": "#ff8c69",
    "muted": "#5f635e",
    "borde": "#a8aaa2",
}

PALETA_OSCURA = {
    "fondo": "#111713",
    "superficie": "#1a241d",
    "texto": "#f7f7f5",
    "borde": "#556158",
}

VISTAS_NO_SUPERVISADAS = {"acp", "kmeans", "kmedoids", "hac"}

VISTAS_CLASIFICADORES = {
    "clasificacion": "RF",
    "clasificacion_knn": "KNN",
    "clasificacion_dt": "DT",
    "clasificacion_xgboost": "XGBoost",
    "clasificacion_adaboost": "AdaBoost",
    "clasificacion_nb": "NR",
}

TITULOS_VISTA = {
    "datos": "Datos y preparación",
    "eda": "Exploración de datos",
    "acp": "ACP",
    "kmeans": "K-Means",
    "kmedoids": "K-Medoids",
    "hac": "Clustering jerárquico",
    "clasificacion": "Random Forest",
    "clasificacion_knn": "KNN",
    "clasificacion_dt": "Árbol de decisión",
    "clasificacion_xgboost": "XGBoost",
    "clasificacion_adaboost": "AdaBoost",
    "clasificacion_nb": "Naive Bayes",
    "clasificacion_configuracion": "Configuración",
    "lab2_experimentos": "Comparar modelos entrenados",
    "lab2_benchmark": "Evaluación Comparativa de Modelos",
    "lab2_resultados": "Resultados de clasificación",
    "regresion": "Regresión",
    "lab3_benchmark": "Evaluación Comparativa de Modelos",
    "lab3_comparacion": "Comparar modelos entrenados",
    "lab3_resultados": "Resultados de regresión",
}

RUTAS_VISTA = {
    "datos": "Datos / Preparación",
    "eda": "Exploración / EDA",
    "acp": "Exploración / ACP",
    "kmeans": "Clustering / K-Means",
    "kmedoids": "Clustering / K-Medoids",
    "hac": "Clustering / HAC",
    "clasificacion": "Clasificación / Random Forest",
    "clasificacion_knn": "Clasificación / KNN",
    "clasificacion_dt": "Clasificación / Árbol de decisión",
    "clasificacion_xgboost": "Clasificación / XGBoost",
    "clasificacion_adaboost": "Clasificación / AdaBoost",
    "clasificacion_nb": "Clasificación / Naive Bayes",
    "clasificacion_configuracion": "Configuración / Clasificación",
    "lab2_experimentos": "Clasificación / Comparar modelos entrenados",
    "lab2_benchmark": "Clasificación / Evaluación Comparativa de Modelos",
    "lab2_resultados": "Resultados / Clasificación / Validación y prueba",
    "regresion": "Regresión / Modelos",
    "lab3_benchmark": "Regresión / Evaluación Comparativa de Modelos",
    "lab3_comparacion": "Regresión / Comparar modelos entrenados",
    "lab3_resultados": "Resultados / Regresión / Validación y prueba",
}

TAREAS = {"clasificacion": "Clasificación", "regresion": "Regresión"}

OPCIONES_COMUNES = (
    "target",
    "features",
    "test_size",
    "random_state",
    "incluir_categoricas",
    "imputar",
    "escalar",
    "usar_particion_global",
)

FOLDS_LAB2 = 5
ESTRATEGIAS = {
    "cv": f"Validación cruzada estratificada ({FOLDS_LAB2} pliegues)",
    "holdout": "Validación simple (una partición)",
}
TOTAL_CANDIDATOS = sum(
    len(lista) for lista in configuraciones_experimento().values()
)
