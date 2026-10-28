"""Internationalization (i18n) engine for Nexo AI / Framework IA.

Supports dynamic language switching between Spanish ('es') and English ('en').
"""

from __future__ import annotations

import streamlit as st
from typing import Any

IDIOMAS = {
    "es": "Español",
    "en": "English",
}

TEXTOS = {
    "es": {
        # App & Header
        "app_nombre": "Nexo IA",
        "app_subtitulo": "Framework modular para análisis exploratorio y aprendizaje automático",
        "idioma_label": "Idioma",
        "actualizar_graficos": "Actualizar gráficos",
        "actualizar_graficos_help": "Úselo después de cambiar el tema en Configuración para actualizar los colores de gráficos y estilos.",
        
        # Navigation
        "nav_seccion": "Navegación",
        "nav_datos": "Datos y preparación",
        "nav_exploracion": "Exploración de datos",
        "nav_eda": "EDA",
        "nav_acp": "ACP",
        "nav_analisis": "Análisis y modelos",
        "nav_configuracion": "Configuración",
        "nav_clustering": "Clustering",
        "nav_kmeans": "K-Means",
        "nav_kmedoids": "K-Medoids",
        "nav_hac": "HAC",
        "nav_clustering_jerarquico": "Clustering jerárquico",
        "nav_clasificacion": "Clasificación",
        "nav_modelos_disponibles": "Modelos disponibles",
        "nav_knn": "KNN",
        "nav_dt": "Árbol de decisión",
        "nav_rf": "Random Forest",
        "nav_xgboost": "XGBoost",
        "nav_adaboost": "AdaBoost",
        "nav_nb": "Naive Bayes",
        "nav_benchmark": "Evaluación Comparativa",
        "nav_benchmark_completo": "Evaluación Comparativa de Modelos",
        "nav_regresion": "Regresión",
        "nav_reg_dt": "Árbol de regresión",
        "nav_reg_rf": "Random Forest Regressor",
        "nav_reg_rls": "Regresión lineal simple",
        "nav_reg_rlm": "Regresión lineal múltiple",
        "nav_reg_lasso": "Lasso",
        "nav_reg_lassocv": "LassoCV",
        "nav_reg_ridge": "Ridge",
        "nav_reg_ridgecv": "RidgeCV",
        "nav_reg_svr": "SVR · vectores de soporte",
        "nav_resultados": "Resultados",
        "nav_resultados_clasificacion": "Resultados de clasificación",
        "nav_resultados_regresion": "Resultados de regresión",
        "nav_resultados_help": "Ejecute primero un modelo o evaluación comparativa.",
        
        # Source & Data
        "fuente_titulo": "Fuente de datos",
        "fuente_aplicar": "Aplicar fuente",
        "fuente_preset": "Preajuste de datos",
        "fuente_subir": "Subir archivo CSV",
        "fuente_separador": "Separador",
        "fuente_pendiente_titulo": "Etapa 1 · Fuente pendiente",
        "fuente_pendiente_desc": "Aplique un CSV o la referencia local desde Fuente de datos para habilitar el análisis.",
        "fuente_pendiente_info": "Seleccione un CSV local o suba un archivo para comenzar.",
        "calidad_datos": "Calidad de datos",
        "resumen_filas": "Filas",
        "resumen_columnas": "Columnas",
        "resumen_nulos": "Valores nulos",
        "resumen_duplicados": "Duplicados",
        
        # Results View
        "res_modelos_ind": "Modelos individuales",
        "res_eval_comp": "Evaluación comparativa",
        "res_sin_modelos_ind": "No se han entrenado modelos individuales.",
        "res_sin_benchmark": "No se ha ejecutado la evaluación comparativa.",
        "res_ver_hiperparametros": "Ver hiperparámetros de los modelos",
        "res_sin_hiperparametros": "Sin hiperparámetros registrados.",
        "res_parametro": "Parámetro",
        "res_valor": "Valor",
        "res_modelo_ganador": "Modelo ganador",
        "res_metricas_val": "Métricas en validación cruzada",
        "res_metricas_test": "Métricas en conjunto de prueba",
        "res_ir_a_resultados": "Ir a la pantalla de Resultados",
        "res_ejecutar_evaluacion": "Ejecutar evaluación comparativa",
        "res_entrenando": "Entrenando modelo...",
        "res_evaluando": "Ejecutando evaluación comparativa...",
        
        # Supervised Configuration
        "cfg_tarea": "Tarea supervisada",
        "cfg_target": "Variable objetivo (Target)",
        "cfg_features": "Variables predictoras (Features)",
        "cfg_test_size": "Proporción del conjunto de prueba",
        "cfg_random_state": "Semilla aleatoria (Random State)",
        "cfg_escalar": "Escalar características numéricas",
        "cfg_imputar": "Imputar valores faltantes",
        "cfg_estratificar": "Partición estratificada",
        # General & Breadcrumbs
        "dataset_activo": "Dataset activo",
        "configurar_tarea": "Configurar tarea",
        "flujo_continuar_modelo": "Continuar a un modelo",
        "flujo_revisar_config": "Revisar configuración",
        "flujo_revisar_resultados": "Revisar resultados guardados",
        
        # Metrics
        "metric_filas": "Filas",
        "metric_columnas": "Columnas",
        "metric_duplicados": "Duplicados",
        "metric_nulos": "Nulos",
        "metric_porcentaje_nulos": "Porcentaje Nulos",
        "metric_numericas": "Numéricas",
        "metric_categoricas": "Categóricas",
        "metric_outliers_criticos": "Outliers Críticos",

        # Data Preparation
        "prep_dataset": "Preparación del dataset",
        "prep_tipos_columna": "Tipos de columna",
        "prep_limpieza": "Preparación y limpieza",
        "prep_columnas_eliminar": "Columnas a eliminar",
        "prep_columnas_eliminar_help": "Se descartan del dataset original antes de aplicar el resto de la preparación.",
        "prep_filas_sin_nulos": "Eliminar filas con nulos en estas columnas",
        "prep_filas_sin_nulos_help": "Se elimina la fila completa si falta un valor en alguna columna seleccionada. El objetivo sólo se filtra si lo selecciona; no se rellena ni se deduce automáticamente.",
        "prep_eliminar_duplicados": "Eliminar duplicados",
        "prep_transf_globales": "Transformaciones globales · solo exploración y clustering",
        "prep_imputar_nulos": "Imputar valores nulos",
        "prep_escalado": "Escalado permanente",
        "prep_escalado_ninguno": "Ninguno",
        "prep_escalado_normalizar": "Normalizar",
        "prep_escalado_estandarizar": "Estandarizar",
        "prep_aplicar": "Aplicar preparación",
        "prep_restaurar": "Restaurar dataset original",
        "prep_descargar": "Descargar dataset actual",
        "prep_division_titulo": "División en entrenamiento y prueba",
        "prep_calcular_particion": "Calcular partición",

        # Source
        "fuente_origen_label": "Origen",
        "fuente_csv_local": "CSV local",
        "fuente_subir_csv": "Subir CSV",
        "fuente_diabetes_ref": "Diabetes (referencia)",
        "fuente_separador_decimal": "Separador decimal",

        # Supervised Configuration
        "cfg_distribucion_target": "Distribución del objetivo",
        "cfg_feature_simple": "Predictor de regresión lineal simple",
        "cfg_particion_reproducibilidad": "Partición y reproducibilidad",
        "cfg_usar_particion": "Usar partición calculada",
        "cfg_prep_entrenamiento": "Preparación aprendida solo con entrenamiento",
        "cfg_codificar_cat": "Codificar categóricas",
        "cfg_analisis_titulo": "Configuración del análisis",
    },
    "en": {
        # App & Header
        "app_nombre": "Nexo AI",
        "app_subtitulo": "Modular framework for exploratory data analysis and machine learning",
        "idioma_label": "Language",
        "actualizar_graficos": "Refresh Charts",
        "actualizar_graficos_help": "Use after changing the theme in Settings to refresh chart colors and styles.",
        
        # Navigation
        "nav_seccion": "Navigation",
        "nav_datos": "Data & Preparation",
        "nav_exploracion": "Data Exploration",
        "nav_eda": "EDA",
        "nav_acp": "PCA",
        "nav_analisis": "Analysis & Models",
        "nav_configuracion": "Settings",
        "nav_clustering": "Clustering",
        "nav_kmeans": "K-Means",
        "nav_kmedoids": "K-Medoids",
        "nav_hac": "HAC",
        "nav_clustering_jerarquico": "Hierarchical Clustering",
        "nav_clasificacion": "Classification",
        "nav_modelos_disponibles": "Available Models",
        "nav_knn": "KNN",
        "nav_dt": "Decision Tree",
        "nav_rf": "Random Forest",
        "nav_xgboost": "XGBoost",
        "nav_adaboost": "AdaBoost",
        "nav_nb": "Naive Bayes",
        "nav_benchmark": "Comparative Evaluation",
        "nav_benchmark_completo": "Model Comparative Evaluation",
        "nav_regresion": "Regression",
        "nav_reg_dt": "Regression Tree",
        "nav_reg_rf": "Random Forest Regressor",
        "nav_reg_rls": "Simple Linear Regression",
        "nav_reg_rlm": "Multiple Linear Regression",
        "nav_reg_lasso": "Lasso",
        "nav_reg_lassocv": "LassoCV",
        "nav_reg_ridge": "Ridge",
        "nav_reg_ridgecv": "RidgeCV",
        "nav_reg_svr": "SVR · Support Vector Regressor",
        "nav_resultados": "Results",
        "nav_resultados_clasificacion": "Classification Results",
        "nav_resultados_regresion": "Regression Results",
        "nav_resultados_help": "Run a model or benchmark evaluation first.",
        
        # Source & Data
        "fuente_titulo": "Data Source",
        "fuente_aplicar": "Apply Source",
        "fuente_preset": "Dataset Preset",
        "fuente_subir": "Upload CSV File",
        "fuente_separador": "Delimiter",
        "fuente_pendiente_titulo": "Stage 1 · Pending Data Source",
        "fuente_pendiente_desc": "Apply a CSV or the local reference from Data Source to enable analysis.",
        "fuente_pendiente_info": "Select a local CSV or upload a file to begin.",
        "calidad_datos": "Data Quality",
        "resumen_filas": "Rows",
        "resumen_columnas": "Columns",
        "resumen_nulos": "Missing Values",
        "resumen_duplicados": "Duplicates",
        
        # Results View
        "res_modelos_ind": "Individual Models",
        "res_eval_comp": "Comparative Evaluation",
        "res_sin_modelos_ind": "No individual models have been trained yet.",
        "res_sin_benchmark": "Comparative benchmark has not been run yet.",
        "res_ver_hiperparametros": "View Model Hyperparameters",
        "res_sin_hiperparametros": "No registered hyperparameters.",
        "res_parametro": "Parameter",
        "res_valor": "Value",
        "res_modelo_ganador": "Winning Model",
        "res_metricas_val": "Cross-Validation Metrics",
        "res_metricas_test": "Test Set Metrics",
        "res_ir_a_resultados": "Go to Results Page",
        "res_ejecutar_evaluacion": "Run Comparative Evaluation",
        "res_entrenando": "Training model...",
        "res_evaluando": "Running comparative evaluation...",
        
        # Supervised Configuration
        "cfg_tarea": "Supervised Task",
        "cfg_target": "Target Variable",
        "cfg_features": "Predictor Features",
        "cfg_test_size": "Test Split Ratio",
        "cfg_random_state": "Random Seed",
        "cfg_escalar": "Scale Numerical Features",
        "cfg_imputar": "Impute Missing Values",
        "cfg_estratificar": "Stratified Split",
        # General & Breadcrumbs
        "dataset_activo": "Active dataset",
        "configurar_tarea": "Configure task",
        "flujo_continuar_modelo": "Continue to a model",
        "flujo_revisar_config": "Review configuration",
        "flujo_revisar_resultados": "Review saved results",
        
        # Metrics
        "metric_filas": "Rows",
        "metric_columnas": "Columns",
        "metric_duplicados": "Duplicates",
        "metric_nulos": "Nulls",
        "metric_porcentaje_nulos": "% Nulls",
        "metric_numericas": "Numerical",
        "metric_categoricas": "Categorical",
        "metric_outliers_criticos": "Critical Outliers",

        # Data Preparation
        "prep_dataset": "Dataset preparation",
        "prep_tipos_columna": "Column types",
        "prep_limpieza": "Preparation & cleaning",
        "prep_columnas_eliminar": "Columns to remove",
        "prep_columnas_eliminar_help": "Discarded from the original dataset before applying the rest of preparation.",
        "prep_filas_sin_nulos": "Drop rows with nulls in these columns",
        "prep_filas_sin_nulos_help": "The entire row is dropped if a value is missing in any selected column.",
        "prep_eliminar_duplicados": "Remove duplicates",
        "prep_transf_globales": "Global transformations · exploration and clustering only",
        "prep_imputar_nulos": "Impute missing values",
        "prep_escalado": "Permanent scaling",
        "prep_escalado_ninguno": "None",
        "prep_escalado_normalizar": "Normalize",
        "prep_escalado_estandarizar": "Standardize",
        "prep_aplicar": "Apply preparation",
        "prep_restaurar": "Restore original dataset",
        "prep_descargar": "Download current dataset",
        "prep_division_titulo": "Train & test split",
        "prep_calcular_particion": "Compute partition",

        # Source
        "fuente_origen_label": "Source",
        "fuente_csv_local": "Local CSV",
        "fuente_subir_csv": "Upload CSV",
        "fuente_diabetes_ref": "Diabetes (reference)",
        "fuente_separador_decimal": "Decimal separator",

        # Supervised Configuration
        "cfg_distribucion_target": "Target Distribution",
        "cfg_feature_simple": "Simple linear regression predictor",
        "cfg_particion_reproducibilidad": "Split & reproducibility",
        "cfg_usar_particion": "Use precomputed split",
        "cfg_prep_entrenamiento": "Training-only learned preprocessing",
        "cfg_codificar_cat": "Encode categorical features",
        "cfg_analisis_titulo": "Analysis Settings",
    },
}


def obtener_idioma() -> str:
    """Devuelve el código de idioma activo ('es' o 'en')."""
    return st.session_state.get("idioma", "es")


def cambiar_idioma(nuevo_idioma: str) -> None:
    """Establece el idioma en la sesión."""
    if nuevo_idioma in IDIOMAS:
        st.session_state["idioma"] = nuevo_idioma


def t(clave: str, fallback: str | None = None, **kwargs: Any) -> str:
    """Traduce una clave según el idioma activo."""
    lang = obtener_idioma()
    texto = TEXTOS.get(lang, {}).get(clave)
    if texto is None:
        texto = TEXTOS.get("es", {}).get(clave)
    if texto is None:
        texto = fallback if fallback is not None else clave
    if kwargs:
        try:
            return texto.format(**kwargs)
        except Exception:
            return texto
    return texto


def render_selector_idioma() -> None:
    """Renderiza el conmutador de idioma en la barra lateral."""
    idioma_actual = obtener_idioma()
    opciones = list(IDIOMAS.keys())
    indice = opciones.index(idioma_actual) if idioma_actual in opciones else 0
    
    nuevo_idioma = st.selectbox(
        t("idioma_label"),
        options=opciones,
        index=indice,
        format_func=lambda codigo: IDIOMAS.get(codigo, codigo),
        key="selector_idioma_global",
    )
    if nuevo_idioma != idioma_actual:
        cambiar_idioma(nuevo_idioma)
        st.rerun()
