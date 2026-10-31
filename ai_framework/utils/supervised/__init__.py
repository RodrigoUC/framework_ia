"""Utilities for supervised models and workflows (classification and regression)."""

from __future__ import annotations

import sys
from . import (
    classification,
    classification_config,
    comparison,
    cross_validation,
    evidence,
    experiments,
    export,
    regression,
    regression_algorithms,
    regression_config,
)

algoritmos_regresion = regression_algorithms
clasificacion = classification
comparacion = comparison
configuracion_clasificacion = classification_config
configuracion_regresion = regression_config
evidencia = evidence
experimentos = experiments
exportacion = export
regresion = regression
validacion_cruzada = cross_validation

sys.modules.setdefault("ai_framework.utils.supervised.algoritmos_regresion", regression_algorithms)
sys.modules.setdefault("ai_framework.utils.supervised.clasificacion", classification)
sys.modules.setdefault("ai_framework.utils.supervised.comparacion", comparison)
sys.modules.setdefault("ai_framework.utils.supervised.configuracion_clasificacion", classification_config)
sys.modules.setdefault("ai_framework.utils.supervised.configuracion_regresion", regression_config)
sys.modules.setdefault("ai_framework.utils.supervised.evidencia", evidence)
sys.modules.setdefault("ai_framework.utils.supervised.experimentos", experiments)
sys.modules.setdefault("ai_framework.utils.supervised.exportacion", export)
sys.modules.setdefault("ai_framework.utils.supervised.regresion", regression)
sys.modules.setdefault("ai_framework.utils.supervised.validacion_cruzada", cross_validation)
from .regression_algorithms import (
    crear_modelo as crear_modelo_regresion,
    metadatos_regresion,
    normalizar_algoritmo as normalizar_algoritmo_regresion,
    parametros_regresion,
)
from .classification import (
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
from .comparison import (
    comparar_algoritmos_supervisados,
    despachar_algoritmo_supervisado,
)
from .classification_config import (
    ALGORITMOS_CLASIFICACION,
    CONFIGURACIONES_MODELO,
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
    ConfiguracionModelo,
    ParametroModelo,
    configuraciones_experimento,
)
from .regression_config import (
    ALGORITMOS_LAB3,
    ALGORITMOS_REGRESION,
    ALPHAS_LAB3,
    ALPHAS_REGRESION,
    CONFIGURACIONES_REGRESION,
    ConfiguracionRegresion,
    NOMBRES_REGRESORES,
    configuraciones_experimento_regresion,
)
from .experiments import (
    ExperimentoClasificacion,
    ExperimentoRegresion,
    ExperimentoSupervisado,
    ejecutar_experimento,
    ejecutar_experimento_clasificacion,
    ejecutar_experimento_regresion,
)
from .regression import (
    ajustar_regresion,
    crear_resultado_regresion,
    metricas_regresion,
    validar_datos_regresion,
    validar_objetivo as validar_objetivo_regresion,
    validar_predictoras as validar_predictoras_regresion,
)
from .cross_validation import ejecutar_experimento_cv

algoritmos_regresion = regression_algorithms
experimentos = experiments
validacion_cruzada = cross_validation
evidencia = evidence
exportacion = export

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
    "comparar_algoritmos_supervisados",
    "despachar_algoritmo_supervisado",
    "ALGORITMOS_CLASIFICACION",
    "CONFIGURACIONES_MODELO",
    "METRICAS_SELECCION",
    "NOMBRES_MODELOS",
    "ConfiguracionModelo",
    "ParametroModelo",
    "configuraciones_experimento",
    "ALGORITMOS_LAB3",
    "ALGORITMOS_REGRESION",
    "ALPHAS_LAB3",
    "ALPHAS_REGRESION",
    "CONFIGURACIONES_REGRESION",
    "ConfiguracionRegresion",
    "NOMBRES_REGRESORES",
    "configuraciones_experimento_regresion",
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
    "regression_algorithms",
    "experiments",
    "cross_validation",
    "evidence",
    "export",
    "algoritmos_regresion",
    "experimentos",
    "validacion_cruzada",
    "evidencia",
    "exportacion",
]
