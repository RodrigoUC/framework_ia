"""Funciones y objetos de configuración compartidos por el framework."""

from .cache_numba import configurar_cache_numba
from .colores import color_texto_por_luminancia
from .compatibilidad_sklearn import crear_codificador_denso, crear_tsne
from .configuracion_clasificacion import (
    ALGORITMOS_CLASIFICACION,
    ALGORITMOS_LAB2,
    CONFIGURACIONES_MODELO,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
    configuraciones_experimento,
)
from .configuracion_csv import ConfiguracionCSV
from .configuracion_particion import ConfiguracionParticion
from .configuracion_preprocesamiento import ConfiguracionPreprocesamiento
from .configuracion_regresion import (
    ALGORITMOS_LAB3,
    ALGORITMOS_REGRESION,
    ALPHAS_LAB3,
    ALPHAS_REGRESION,
    CONFIGURACIONES_REGRESION,
    NOMBRES_REGRESORES,
    configuraciones_experimento_regresion,
)
from .evaluacion_cluster import (
    calcular_centroides,
    construir_resultado_cluster,
    evaluar_hac,
    evaluar_kmeans,
    validar_numero_clusters,
)
from .k_medoids import ModeloKMedoids
from .no_supervisado import (
    ALGORITMOS_CLUSTERING_COMPARACION,
    comparar_algoritmos_clustering,
    despachar_algoritmo,
)
from .supervisado import (
    algoritmos_regresion,
    clasificacion as utils_clasificacion,
    experimentos as utils_experimentos,
    regresion as utils_regresion,
    validacion_cruzada as utils_validacion_cruzada,
)

__all__ = [
    "ALGORITMOS_CLASIFICACION",
    "ALGORITMOS_CLUSTERING_COMPARACION",
    "ALGORITMOS_LAB2",
    "ALGORITMOS_LAB3",
    "ALGORITMOS_REGRESION",
    "ALPHAS_LAB3",
    "ALPHAS_REGRESION",
    "CONFIGURACIONES_MODELO",
    "CONFIGURACIONES_REGRESION",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "METRICAS_SELECCION",
    "ModeloKMedoids",
    "NOMBRES_MODELOS",
    "NOMBRES_REGRESORES",
    "calcular_centroides",
    "color_texto_por_luminancia",
    "comparar_algoritmos_clustering",
    "configuraciones_experimento",
    "configuraciones_experimento_regresion",
    "configurar_cache_numba",
    "construir_resultado_cluster",
    "crear_codificador_denso",
    "crear_tsne",
    "despachar_algoritmo",
    "evaluar_hac",
    "evaluar_kmeans",
    "validar_numero_clusters",
]
