"""Capa de controladores de aplicación para la interfaz Streamlit."""

from ai_framework.ui.controllers import (
    classification_controller,
    regression_controller,
    sources_controller,
    unsupervised_controller,
)

controlador_clasificacion = classification_controller
controlador_regresion = regression_controller
controlador_fuentes = sources_controller
controlador_no_supervisado = unsupervised_controller

from ai_framework.ui.controllers.classification_controller import (
    _argumentos_comunes,
    _argumentos_exportacion,
    _contexto_ejecucion,
    _fuente_ejecutada,
    _json,
    _metricas_por_clase,
    _metricas_porcentuales,
    _modelo,
    nombre_tasa_verdaderos,
)
from ai_framework.ui.controllers.sources_controller import (
    _aplicar_configuracion_diabetes,
    _aplicar_preset_regresion,
    _cargar_fuente_csv,
    _cargar_referencia_regresion,
    _guardar_metadatos_fuente,
    _limpiar_resultados,
    _sincronizar_dataset,
    _usar_particion,
)
from ai_framework.ui.controllers.unsupervised_controller import (
    _columnas_categoricas,
    _columnas_numericas,
    _crear_cluster,
    _crear_reductor,
    _muestrear,
    comparar_metodos_hac,
    ejecutar_acp,
    ejecutar_hac,
    ejecutar_kmeans,
    ejecutar_kmedoids,
    ejecutar_tsne,
    ejecutar_umap,
    evaluar_kmeans_k,
)
from ai_framework.ui.controllers.regression_controller import (
    argumentos_comunes,
    crear_regresor,
    snapshot_ejecucion,
)

__all__ = [
    "_aplicar_configuracion_diabetes",
    "_aplicar_preset_regresion",
    "_argumentos_comunes",
    "_argumentos_exportacion",
    "_cargar_fuente_csv",
    "_cargar_referencia_regresion",
    "_columnas_categoricas",
    "_columnas_numericas",
    "_contexto_ejecucion",
    "_crear_cluster",
    "_crear_reductor",
    "_fuente_ejecutada",
    "_guardar_metadatos_fuente",
    "_json",
    "_limpiar_resultados",
    "_metricas_por_clase",
    "_metricas_porcentuales",
    "_modelo",
    "_muestrear",
    "_sincronizar_dataset",
    "_usar_particion",
    "argumentos_comunes",
    "comparar_metodos_hac",
    "crear_regresor",
    "ejecutar_acp",
    "ejecutar_hac",
    "ejecutar_kmeans",
    "ejecutar_kmedoids",
    "ejecutar_tsne",
    "ejecutar_umap",
    "evaluar_kmeans_k",
    "nombre_tasa_verdaderos",
    "snapshot_ejecucion",
]
