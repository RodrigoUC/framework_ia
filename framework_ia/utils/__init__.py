"""Funciones y objetos de configuración compartidos por el framework."""

from .cache_numba import configurar_cache_numba
from .colores import color_texto_por_luminancia
from .compatibilidad_sklearn import crear_codificador_denso, crear_tsne
from .configuracion_agentes import (
    AGENTE_CLUSTERING,
    AGENTE_EDA,
    NOMBRES_PROVEEDORES,
    PROVEEDOR_SIN_LLM,
    PROVEEDORES_LLM,
    UmbralesClustering,
    UmbralesEDA,
)
from .configuracion_clasificacion import (
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
    CONFIGURACIONES_REGRESION,
    NOMBRES_REGRESORES,
    configuraciones_experimento_regresion,
)

__all__ = [
    "AGENTE_CLUSTERING",
    "AGENTE_EDA",
    "ALGORITMOS_LAB2",
    "ALGORITMOS_LAB3",
    "CONFIGURACIONES_MODELO",
    "CONFIGURACIONES_REGRESION",
    "METRICAS_SELECCION",
    "NOMBRES_MODELOS",
    "NOMBRES_PROVEEDORES",
    "PROVEEDOR_SIN_LLM",
    "PROVEEDORES_LLM",
    "NOMBRES_REGRESORES",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "UmbralesClustering",
    "UmbralesEDA",
    "color_texto_por_luminancia",
    "configuraciones_experimento",
    "configuraciones_experimento_regresion",
    "configurar_cache_numba",
    "crear_codificador_denso",
    "crear_tsne",
]
