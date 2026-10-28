"""Funciones y objetos de configuración compartidos por el framework."""

import sys

from .cache_numba import configurar_cache_numba
from .colores import color_texto_por_luminancia
from .compatibilidad_sklearn import crear_codificador_denso, crear_tsne
from .datos import (
    ConfiguracionCSV,
    ConfiguracionParticion,
    ConfiguracionPreprocesamiento,
)
from .no_supervisado import (
    ALGORITMOS_CLUSTERING_COMPARACION,
    ModeloKMedoids,
    calcular_centroides,
    comparar_algoritmos_clustering,
    construir_resultado_cluster,
    despachar_algoritmo,
    evaluar_hac,
    evaluar_kmeans,
    validar_numero_clusters,
)
from .supervisado import (
    ALGORITMOS_CLASIFICACION,
    ALGORITMOS_LAB2,
    ALGORITMOS_LAB3,
    ALGORITMOS_REGRESION,
    ALPHAS_LAB3,
    ALPHAS_REGRESION,
    CONFIGURACIONES_MODELO,
    CONFIGURACIONES_REGRESION,
    ConfiguracionModelo,
    ConfiguracionRegresion,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
    NOMBRES_REGRESORES,
    ParametroModelo,
    algoritmos_regresion,
    clasificacion as utils_clasificacion,
    comparar_algoritmos_supervisados,
    configuraciones_experimento,
    configuraciones_experimento_regresion,
    despachar_algoritmo_supervisado,
    experimentos as utils_experimentos,
    regresion as utils_regresion,
    validacion_cruzada as utils_validacion_cruzada,
)
from . import datos, no_supervisado, supervisado

# Alias en sys.modules para mantener compatibilidad transparente
sys.modules.setdefault("framework_ia.utils.k_medoids", no_supervisado.k_medoids)
sys.modules.setdefault("framework_ia.utils.evaluacion_cluster", no_supervisado.evaluacion_cluster)
sys.modules.setdefault("framework_ia.utils.configuracion_clasificacion", supervisado.configuracion_clasificacion)
sys.modules.setdefault("framework_ia.utils.configuracion_regresion", supervisado.configuracion_regresion)
sys.modules.setdefault("framework_ia.utils.configuracion_csv", datos.configuracion_csv)
sys.modules.setdefault("framework_ia.utils.configuracion_particion", datos.configuracion_particion)
sys.modules.setdefault("framework_ia.utils.configuracion_preprocesamiento", datos.configuracion_preprocesamiento)

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
    "ConfiguracionModelo",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "ConfiguracionRegresion",
    "METRICAS_SELECCION",
    "ModeloKMedoids",
    "NOMBRES_MODELOS",
    "NOMBRES_REGRESORES",
    "ParametroModelo",
    "calcular_centroides",
    "color_texto_por_luminancia",
    "comparar_algoritmos_clustering",
    "comparar_algoritmos_supervisados",
    "configuraciones_experimento",
    "configuraciones_experimento_regresion",
    "configurar_cache_numba",
    "construir_resultado_cluster",
    "crear_codificador_denso",
    "crear_tsne",
    "despachar_algoritmo",
    "despachar_algoritmo_supervisado",
    "evaluar_hac",
    "evaluar_kmeans",
    "validar_numero_clusters",
]
