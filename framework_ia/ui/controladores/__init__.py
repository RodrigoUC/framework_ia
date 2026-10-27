"""Capa de controladores de aplicación para la interfaz Streamlit."""

from framework_ia.ui.controladores.controlador_clasificacion import (
    _argumentos_comunes,
    _argumentos_exportacion,
    _contexto_ejecucion,
    _fuente_ejecutada,
    _json,
    _metricas_por_clase,
    _metricas_porcentuales,
    _modelo,
)
from framework_ia.ui.controladores.controlador_fuentes import (
    _aplicar_configuracion_diabetes,
    _aplicar_preset_regresion,
    _cargar_fuente_csv,
    _cargar_referencia_regresion,
    _guardar_metadatos_fuente,
    _limpiar_resultados,
    _sincronizar_dataset,
    _usar_particion,
)
from framework_ia.ui.controladores.controlador_no_supervisado import (
    _columnas_categoricas,
    _columnas_numericas,
    _crear_cluster,
    _crear_reductor,
    _muestrear,
)
from framework_ia.ui.controladores.controlador_regresion import (
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
    "crear_regresor",
    "snapshot_ejecucion",
]
