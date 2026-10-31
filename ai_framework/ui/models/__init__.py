"""Capa de modelos de estado, disponibilidad y catálogo de vistas para la UI."""

from ai_framework.ui.models import (
    agents_state,
    availability,
    classification_state,
    diabetes_profile,
    regression_state,
    session_state,
    view_catalog,
)

catalogo_vistas = view_catalog
disponibilidad = availability
estado_agentes = agents_state
estado_clasificacion = classification_state
estado_regresion = regression_state
estado_sesion = session_state
perfil_diabetes = diabetes_profile

from ai_framework.ui.models.view_catalog import (
    APP_NAME,
    BASE_DIR,
    ESTRATEGIAS,
    FOLDS_LAB2,
    OPCIONES_COMUNES,
    PALETA,
    PALETA_OSCURA,
    RUTAS_VISTA,
    TAREAS,
    TITULOS_VISTA,
    TOTAL_CANDIDATOS,
    VISTAS_CLASIFICADORES,
    VISTAS_NO_SUPERVISADAS,
    VISTAS_REGRESION,
    VISTAS_REGRESORES,
)
from ai_framework.ui.models.agents_state import (
    limpiar_resultados_agentes,
    sincronizar_contexto_agentes,
    sincronizar_contexto_clustering,
)
from ai_framework.ui.models.availability import (
    Disponibilidad,
    disponibilidad_supervisada,
    hay_evidencia,
)
from ai_framework.ui.models.classification_state import (
    configuracion_clasificacion_actual,
    firma_clasificacion,
    limpiar_resultados_clasificacion,
    sincronizar_contexto,
    sincronizar_ejecucion,
)
from ai_framework.ui.models.regression_state import (
    configuracion_regresion_actual,
    firma_regresion,
    limpiar_resultados_regresion,
    sincronizar_contexto_regresion,
)
from ai_framework.ui.models.session_state import (
    _firma,
    _guardar_resultado,
    _recuperar_resultado,
    estado_para_familia,
    firma_parametros,
    firma_preparacion,
    huella_datos,
    inicializar_configuracion,
    state_value,
)
from ai_framework.ui.models.diabetes_profile import (
    RUTA_PROCEDENCIA,
    estado_fuente_diabetes,
)

__all__ = [
    "APP_NAME",
    "BASE_DIR",
    "Disponibilidad",
    "ESTRATEGIAS",
    "FOLDS_LAB2",
    "OPCIONES_COMUNES",
    "PALETA",
    "PALETA_OSCURA",
    "RUTAS_VISTA",
    "RUTA_PROCEDENCIA",
    "TAREAS",
    "TITULOS_VISTA",
    "TOTAL_CANDIDATOS",
    "VISTAS_CLASIFICADORES",
    "VISTAS_NO_SUPERVISADAS",
    "VISTAS_REGRESION",
    "VISTAS_REGRESORES",
    "_firma",
    "_guardar_resultado",
    "_recuperar_resultado",
    "configuracion_clasificacion_actual",
    "configuracion_regresion_actual",
    "disponibilidad_supervisada",
    "estado_fuente_diabetes",
    "estado_para_familia",
    "firma_clasificacion",
    "firma_parametros",
    "firma_preparacion",
    "firma_regresion",
    "hay_evidencia",
    "huella_datos",
    "inicializar_configuracion",
    "limpiar_resultados_agentes",
    "limpiar_resultados_clasificacion",
    "limpiar_resultados_regresion",
    "sincronizar_contexto",
    "sincronizar_contexto_agentes",
    "sincronizar_contexto_clustering",
    "sincronizar_contexto_regresion",
    "sincronizar_ejecucion",
    "state_value",
]
