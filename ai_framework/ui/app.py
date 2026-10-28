"""Punto de entrada y enrutador principal de la interfaz Streamlit."""

from __future__ import annotations

import sys

import streamlit as st
from ai_framework.ui.i18n import t

from ai_framework.datasets.diabetes import cargar_diabetes_regresion
from ai_framework.data.sources import CargadorCSV, ConfiguracionCSV
from ai_framework.ui.controllers.sources_controller import (
    _aplicar_preset_regresion,
    _cargar_fuente_csv,
    _cargar_referencia_regresion,
    _guardar_metadatos_fuente,
    _limpiar_resultados,
    _sincronizar_dataset,
)
from ai_framework.ui.models.view_catalog import (
    APP_NAME,
    BASE_DIR,
    PALETA,
    PALETA_OSCURA,
    RUTAS_VISTA,
    TITULOS_VISTA,
    VISTAS_CLASIFICADORES,
    VISTAS_NO_SUPERVISADAS,
    VISTAS_REGRESION,
    VISTAS_REGRESORES,
)
from ai_framework.ui.models.availability import (
    disponibilidad_supervisada,
    hay_evidencia,
)
from ai_framework.ui.models.classification_state import (
    configuracion_clasificacion_actual,
    firma_clasificacion as _firma_clasificacion,
    limpiar_resultados_clasificacion,
    sincronizar_contexto,
)
from ai_framework.ui.models.regression_state import (
    configuracion_regresion_actual,
    limpiar_resultados_regresion,
    sincronizar_contexto_regresion,
)
from ai_framework.ui.models.session_state import inicializar_configuracion
from ai_framework.ui.views.navigation import (
    _render_encabezado_atlas,
    _resumen_calidad,
    _seleccionar_vista,
    mostrar_guia,
)
from ai_framework.ui.views.theme import (
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
from ai_framework.ui.views.classification_view import (
    preparacion_segura,
    render_modelo_individual,
    render_resultados,
)
from ai_framework.ui.views.configuration_view import (
    _configurar_modelos,
    _render_configuracion_requerida,
    _render_regresion,
    configurar_clasificacion as _configurar_modelos_clasificacion,
    configurar_regresion,
    configurar_supervisado,
)
from ai_framework.ui.views.data_view import (
    _render_dataset,
    _render_particion,
    _seleccionar_fuente,
    render_perfil_diabetes,
)
from ai_framework.ui.views.classification_experiments_view import (
    render_experimentos,
)
from ai_framework.ui.views.regression_experiments_view import (
    render_experimentos_regresion,
)
from ai_framework.ui.views.unsupervised_view import (
    _render_acp,
    _render_eda,
    _render_hac,
    _render_particional,
)
from ai_framework.ui.views.regression_view import (
    preparacion_regresion_segura,
    render_modelo_regresion,
    render_resultados_regresion,
)


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title=APP_NAME,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _aplicar_estilos_atlas()

    with st.sidebar:
        st.button(
            t("actualizar_graficos", "Actualizar gráficos"),
            key="actualizar_graficos",
            help=t(
                "actualizar_graficos_help",
                "Úselo después de cambiar el tema en Configuración para "
                "actualizar los colores de gráficos y estilos.",
            ),
            use_container_width=True,
        )

    _fn = _seleccionar_fuente
    for _mod in ('framework_ia.ui.streamlit_app', 'ai_framework.ui.streamlit_app'):
        if _mod in sys.modules and hasattr(sys.modules[_mod], '_seleccionar_fuente'):
            _candidate = getattr(sys.modules[_mod], '_seleccionar_fuente')
            if _candidate is not _seleccionar_fuente:
                _fn = _candidate
                break
    datos_cargados, etiqueta, identidad = _fn()
    if datos_cargados is None:
        st.markdown(f"# {APP_NAME}")
        st.caption(
            "Etapa 1 · Fuente pendiente. Aplique un CSV o la referencia local desde Fuente de datos para habilitar el análisis."
        )
        st.info("Seleccione un CSV local o suba un archivo para comenzar.")
        return
    _sincronizar_dataset(datos_cargados, identidad)
    _aplicar_preset_regresion(identidad)
    datos = st.session_state["dataset_preparado"]
    inicializar_configuracion(datos, st.session_state)
    configuraciones_actuales = {
        "clasificacion": configuracion_clasificacion_actual(
            datos, st.session_state
        ),
        "regresion": configuracion_regresion_actual(datos, st.session_state),
    }
    for familia, sincronizar, limpiar in (
        ("clasificacion", sincronizar_contexto, limpiar_resultados_clasificacion),
        ("regresion", sincronizar_contexto_regresion, limpiar_resultados_regresion),
    ):
        actual = configuraciones_actuales[familia]
        if actual is None:
            limpiar(st.session_state)
        else:
            sincronizar(st.session_state, datos, actual)
    disponibilidades = {
        familia: disponibilidad_supervisada(
            datos,
            actual,
            familia,
            preparacion=st.session_state.get("dataset_preparacion"),
        )
        for familia, actual in configuraciones_actuales.items()
    }
    if not disponibilidades["regresion"].lista:
        disponibilidades["RLS"] = disponibilidad_supervisada(
            datos,
            configuraciones_actuales["regresion"],
            "regresion",
            algoritmo="RLS",
            preparacion=st.session_state.get("dataset_preparacion"),
        )
    vista = _seleccionar_vista(disponibilidades)
    _render_encabezado_atlas(datos, etiqueta, vista)
    familia = (
        st.session_state.get("sup_tarea")
        if vista == "configuracion"
        else "regresion"
        if vista in VISTAS_REGRESION
        else "clasificacion"
        if vista
        in {
            *VISTAS_CLASIFICADORES,
            "clasificacion_configuracion",
            "lab2_benchmark",
            "lab2_experimentos",
            "lab2_resultados",
        }
        else None
    )
    disponibilidad_vista = disponibilidades.get(familia)
    if (
        vista in {"regresion_rls", "lab3_benchmark"}
        and disponibilidades.get("RLS") is not None
        and disponibilidades["RLS"].lista
    ):
        disponibilidad_vista = disponibilidades["RLS"]
    mostrar_guia(
        vista,
        familia,
        disponibilidad_vista,
        configuraciones_actuales.get(familia),
    )
    if vista in {"configuracion", "lab2_benchmark"}:
        render_perfil_diabetes(datos)
    if vista == "configuracion":
        configurar_supervisado(datos)
        actual = configuraciones_actuales.get(st.session_state.get("sup_tarea"))
        if actual is not None:
            with st.expander("Partición externa opcional"):
                _render_particion(datos, actual)
        return

    configuracion = None
    configuracion_clasif = None
    configuracion_reg = None
    if vista in VISTAS_REGRESION:
        configuracion_reg = (
            configurar_regresion(datos)
            if vista in {"regresion", "regresion_configuracion"}
            else configuracion_regresion_actual(datos, st.session_state)
        )
        if configuracion_reg is None:
            limpiar_resultados_regresion(st.session_state)
        else:
            sincronizar_contexto_regresion(
                st.session_state, datos, configuracion_reg
            )
            if vista in {"regresion", "regresion_configuracion"}:
                preparacion_regresion_segura()
    if vista in VISTAS_NO_SUPERVISADAS:
        configuracion = _configurar_modelos(datos)
    elif vista == "clasificacion_configuracion":
        configuracion_clasif = _configurar_modelos_clasificacion(
            datos, _mostrar_figura
        )
        if configuracion_clasif is not None:
            sincronizar_contexto(
                st.session_state, datos, configuracion_clasif
            )
            preparacion_segura(mostrar_guia=True)
    elif vista in {
        *VISTAS_CLASIFICADORES,
        "lab2_experimentos",
        "lab2_benchmark",
    }:
        configuracion_clasif = configuracion_clasificacion_actual(
            datos, st.session_state
        )
        if configuracion_clasif is not None:
            sincronizar_contexto(
                st.session_state, datos, configuracion_clasif
            )
        else:
            limpiar_resultados_clasificacion(st.session_state)
    elif vista == "lab2_resultados":
        configuracion_clasif = configuracion_clasificacion_actual(
            datos, st.session_state
        )
        if configuracion_clasif is None:
            limpiar_resultados_clasificacion(st.session_state)
        else:
            sincronizar_contexto(
                st.session_state, datos, configuracion_clasif
            )
            firma_legacy = _firma_clasificacion(
                configuracion_clasif,
                st.session_state.get("dataset_identidad"),
                "modelos_entrenados",
            )
            legacy = st.session_state.get(
                "resultado_modelos_clasificacion", {}
            ).get(firma_legacy, {})
            for codigo, entrada in legacy.items():
                if codigo in {"RF", "NR"} and isinstance(entrada, dict):
                    st.session_state.setdefault(
                        f"resultado_lab2_modelo_{codigo}",
                        entrada.get("resultado"),
                    )

    if vista in VISTAS_REGRESION:
        if configuracion_reg is None or not configuracion_reg["features"]:
            st.warning("Seleccione un target y predictores en Configuración.")
        elif vista in VISTAS_REGRESORES:
            render_modelo_regresion(
                datos,
                configuracion_reg,
                _mostrar_figura,
                algoritmo=VISTAS_REGRESORES[vista],
            )
        elif vista == "lab3_benchmark":
            render_experimentos_regresion(
                datos, configuracion_reg, _mostrar_figura
            )
        elif vista in {"lab3_comparacion", "lab3_resultados"}:
            render_resultados_regresion(
                _mostrar_figura,
                incluir_experimentos=vista == "lab3_resultados",
            )
    elif vista == "datos":
        _render_dataset()
    elif vista == "eda":
        _render_eda(datos)
    elif vista == "clasificacion_configuracion":
        if configuracion_clasif is None:
            st.warning(
                "El dataset necesita al menos dos columnas para configurar "
                "clasificación."
            )
    elif vista == "acp":
        assert configuracion is not None
        _render_acp(datos, configuracion)
    elif vista in VISTAS_CLASIFICADORES:
        if configuracion_clasif is None:
            _render_configuracion_requerida()
        elif not configuracion_clasif["features"]:
            _render_configuracion_requerida(sin_features=True)
        else:
            render_modelo_individual(
                datos,
                configuracion_clasif,
                _mostrar_figura,
                algoritmo=VISTAS_CLASIFICADORES[vista],
            )
    elif vista == "lab2_benchmark":
        if configuracion_clasif is None:
            _render_configuracion_requerida()
        elif not configuracion_clasif["features"]:
            _render_configuracion_requerida(sin_features=True)
        else:
            render_experimentos(datos, configuracion_clasif, _mostrar_figura)
    elif vista == "lab2_experimentos":
        if configuracion_clasif is None:
            _render_configuracion_requerida()
        elif not configuracion_clasif["features"]:
            _render_configuracion_requerida(sin_features=True)
        else:
            render_resultados(_mostrar_figura)
    elif vista == "lab2_resultados":
        if configuracion_clasif is None:
            st.info(
                "Los resultados no están disponibles: el dataset activo necesita "
                "al menos una columna objetivo y una columna predictora."
            )
        else:
            render_resultados(_mostrar_figura, incluir_experimentos=True)
    elif vista == "kmeans":
        assert configuracion is not None
        _render_particional(datos, configuracion, "K-Means")
    elif vista == "kmedoids":
        assert configuracion is not None
        _render_particional(datos, configuracion, "K-Medoids")
    elif vista == "hac":
        assert configuracion is not None
        _render_hac(datos, configuracion)
    else:
        _render_regresion()


if __name__ == "__main__":
    main()
