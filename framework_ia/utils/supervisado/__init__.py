"""Utilidades para modelos y flujos supervisados (clasificación y regresión)."""

from .algoritmos_regresion import (
    crear_modelo as crear_modelo_regresion,
    metadatos_regresion,
    normalizar_algoritmo as normalizar_algoritmo_regresion,
    parametros_regresion,
)
from .clasificacion import (
    ALGORITMOS_LAB2,
    ajustar_clasificacion,
    configuraciones_lab2,
    crear_modelo as crear_modelo_clasificacion,
    crear_resultado_clasificacion,
    metricas_clasificacion,
    normalizar_algoritmo as normalizar_algoritmo_clasificacion,
    ordenar_labels,
    validar_datos_clasificacion,
)
from .experimentos import (
    ExperimentoClasificacion,
    ExperimentoRegresion,
    ExperimentoSupervisado,
    ejecutar_experimento,
    ejecutar_experimento_clasificacion,
    ejecutar_experimento_regresion,
)
from .regresion import (
    ajustar_regresion,
    crear_resultado_regresion,
    metricas_regresion,
    validar_datos_regresion,
    validar_objetivo as validar_objetivo_regresion,
    validar_predictoras as validar_predictoras_regresion,
)
from .validacion_cruzada import ejecutar_experimento_cv

__all__ = [
    "crear_modelo_regresion",
    "metadatos_regresion",
    "normalizar_algoritmo_regresion",
    "parametros_regresion",
    "ALGORITMOS_LAB2",
    "ajustar_clasificacion",
    "configuraciones_lab2",
    "crear_modelo_clasificacion",
    "crear_resultado_clasificacion",
    "metricas_clasificacion",
    "normalizar_algoritmo_clasificacion",
    "ordenar_labels",
    "validar_datos_clasificacion",
    "ExperimentoSupervisado",
    "ExperimentoClasificacion",
    "ExperimentoRegresion",
    "ejecutar_experimento",
    "ejecutar_experimento_clasificacion",
    "ejecutar_experimento_regresion",
    "ajustar_regresion",
    "crear_resultado_regresion",
    "metricas_regresion",
    "validar_datos_regresion",
    "validar_objetivo_regresion",
    "validar_predictoras_regresion",
    "ejecutar_experimento_cv",
]
