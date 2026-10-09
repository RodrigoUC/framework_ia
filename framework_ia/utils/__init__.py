"""Funciones y objetos de configuración compartidos por el framework."""

from .cache_numba import configurar_cache_numba
from .colores import color_texto_por_luminancia
from .compatibilidad_sklearn import crear_codificador_denso, crear_tsne
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

__all__ = [
    "ALGORITMOS_LAB2",
    "CONFIGURACIONES_MODELO",
    "METRICAS_SELECCION",
    "NOMBRES_MODELOS",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "color_texto_por_luminancia",
    "configuraciones_experimento",
    "configurar_cache_numba",
    "crear_codificador_denso",
    "crear_tsne",
]
