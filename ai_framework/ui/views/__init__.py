"""Capa de vistas de presentación para la interfaz Streamlit."""

from ai_framework.ui.views import (
    agents_view,
    classification_experiments_view,
    classification_view,
    configuration_view,
    controls_view,
    data_view,
    navigation,
    regression_experiments_view,
    regression_view,
    results_view,
    theme,
    unsupervised_view,
)

navegacion = navigation
tema = theme
vista_agentes = agents_view
vista_clasificacion = classification_view
vista_configuracion = configuration_view
vista_controles = controls_view
vista_datos = data_view
vista_experimentos_clasificacion = classification_experiments_view
vista_experimentos_regresion = regression_experiments_view
vista_no_supervisado = unsupervised_view
vista_regresion = regression_view
vista_resultados = results_view

from ai_framework.ui.views.agents_view import (
    render_agente_clustering,
    render_agente_eda,
)
from ai_framework.ui.views.navigation import (
    _render_encabezado_atlas,
    _resumen_calidad,
    _seleccionar_vista,
    ir_a,
    mostrar_faltantes,
    mostrar_guia,
)
from ai_framework.ui.views.theme import (
    MostrarFigura,
    _obtener_pestanas,
    render_tabla_clave_valor,
    _ajustar_color_traza,
    _aplicar_estilos_atlas,
    _aplicar_tema_figura,
    _color_rgb,
    _color_texto_celda,
    _color_texto_sector,
    _luminancia,
    _mostrar_figura,
    _texto_mapa_calor,
)
from ai_framework.ui.views.results_view import (
    render_layout_resultados,
    render_panel_modelos_individuales,
)
from ai_framework.ui.views.classification_view import (
    _diagnostico,
    preparacion_segura,
    render_modelo_individual,
    render_resultados,
)
from ai_framework.ui.views.configuration_view import (
    _configurar_modelos,
    _render_configuracion_requerida,
    _render_regresion,
    configurar_clasificacion,
    configurar_regresion,
    configurar_supervisado,
)
from ai_framework.ui.views.controls_view import (
    METRICAS_SELECCION,
    NOMBRES_MODELOS,
    _mostrar_dataframe,
    _seleccionar_entero,
    parametros_individuales,
    parametros_regresion,
)
from ai_framework.ui.views.data_view import (
    _render_dataset,
    _render_particion,
    _resumen_dataset_modulo,
    _seleccionar_fuente,
    render_perfil_diabetes,
)
from ai_framework.ui.views.classification_experiments_view import (
    _descargas,
    _tabla_validacion,
    render_experimentos,
    render_resultado_experimento,
)
from ai_framework.ui.views.regression_experiments_view import (
    mostrar_experimento_guardado,
    render_experimentos_regresion,
)
from ai_framework.ui.views.unsupervised_view import (
    _render_acp,
    _render_eda,
    _render_hac,
    _render_particional,
    _render_tsne,
    _render_umap,
)
from ai_framework.ui.views.regression_view import (
    mostrar_resultado_modelo,
    preparacion_regresion_segura,
    render_modelo_regresion,
    render_resultados_regresion,
)

__all__ = [
    "METRICAS_SELECCION",
    "MostrarFigura",
    "NOMBRES_MODELOS",
    "_ajustar_color_traza",
    "_aplicar_estilos_atlas",
    "_aplicar_tema_figura",
    "_color_rgb",
    "_color_texto_celda",
    "_color_texto_sector",
    "_configurar_modelos",
    "_descargas",
    "_diagnostico",
    "_luminancia",
    "_mostrar_dataframe",
    "_mostrar_figura",
    "_render_acp",
    "_render_configuracion_requerida",
    "_render_dataset",
    "_render_eda",
    "_render_encabezado_atlas",
    "_render_hac",
    "_render_particion",
    "_render_particional",
    "_render_regresion",
    "_render_tsne",
    "_render_umap",
    "_resumen_calidad",
    "_resumen_dataset_modulo",
    "_seleccionar_entero",
    "_seleccionar_fuente",
    "_seleccionar_vista",
    "_tabla_validacion",
    "_texto_mapa_calor",
    "configurar_clasificacion",
    "configurar_regresion",
    "configurar_supervisado",
    "ir_a",
    "mostrar_experimento_guardado",
    "mostrar_faltantes",
    "mostrar_guia",
    "mostrar_resultado_modelo",
    "parametros_individuales",
    "parametros_regresion",
    "preparacion_regresion_segura",
    "preparacion_segura",
    "render_agente_clustering",
    "render_agente_eda",
    "render_experimentos",
    "render_experimentos_regresion",
    "render_modelo_individual",
    "render_modelo_regresion",
    "render_perfil_diabetes",
    "render_resultado_experimento",
    "render_layout_resultados",
    "render_panel_modelos_individuales",
    "render_resultados",
    "render_resultados_regresion",
]
