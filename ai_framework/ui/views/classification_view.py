"""Vistas de clasificación individual, diagnóstico y comparación de resultados."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_framework.ui.controllers.classification_controller import (
    _argumentos_comunes,
    _contexto_ejecucion,
    _json,
    _metricas_por_clase,
    _metricas_porcentuales,
    _modelo,
    nombre_tasa_verdaderos,
)
from ai_framework.ui.models.availability import disponibilidad_supervisada
from ai_framework.ui.models.session_state import firma_parametros
from ai_framework.ui.views.navigation import mostrar_faltantes
from ai_framework.ui.views.theme import MostrarFigura, _obtener_pestanas, render_tabla_clave_valor
from ai_framework.ui.views.controls_view import (
    NOMBRES_MODELOS,
    _mostrar_dataframe,
    parametros_individuales,
)
from ai_framework.utils.visualization import VisualizadorSupervisado


def preparacion_segura(*, mostrar_guia: bool = False) -> bool:
    """Verifica que la preparación previa no haya imputado ni escalado antes del split."""
    preparacion = st.session_state.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        st.error(
            "La preparación actual imputó o escaló antes de separar los datos. "
            "Para evitar fuga de información, desactive esas transformaciones en "
            "Datos y preparación y vuelva a aplicar sus opciones de limpieza. "
            "Ajuste imputación y escalado en Configuración para aprenderlos solo con entrenamiento."
        )
        return False
    if mostrar_guia:
        st.caption(
            "Imputación, codificación y escalado se ajustan automáticamente con el conjunto de entrenamiento. "
            "Las filas con target vacío se excluyen del modelado."
        )
    return True


def render_modelo_individual(
    datos: pd.DataFrame,
    configuracion: dict,
    mostrar_figura: MostrarFigura,
    *,
    algoritmo: str,
) -> None:
    """Entrenamiento exploratorio de un único modelo con parámetros visibles."""
    nombre_modelo = NOMBRES_MODELOS.get(algoritmo, "Naive Bayes")
    segura = preparacion_segura()
    parametros = parametros_individuales(algoritmo)
    disponibilidad = disponibilidad_supervisada(
        datos,
        configuracion,
        "clasificacion",
        algoritmo=algoritmo,
        parametros=parametros,
        preparacion=st.session_state.get("dataset_preparacion"),
    )
    mostrar_faltantes(disponibilidad)
    firma = firma_parametros({"algoritmo": algoritmo, "parametros": parametros})
    estado_modelo = f"modelo_{algoritmo}"
    resultado_modelo = f"resultado_lab2_{estado_modelo}"
    criterio_rf = (
        parametros.get("criterion", "gini") if algoritmo == "RF" else None
    )
    clave_criterios_rf = "resultado_lab2_modelo_RF_criterios"
    intentos_rf = "errores_lab2_modelo_RF"
    if (
        st.button(
            "Entrenar modelo",
            key=f"lab2_entrenar_{algoritmo}",
            type="primary",
            disabled=not configuracion["features"]
            or not segura
            or not disponibilidad.lista,
        )
        and disponibilidad.lista
    ):
        try:
            with st.spinner(f"Entrenando {nombre_modelo}…"):
                resultado = _modelo(datos, configuracion).entrenar(
                    algoritmo=algoritmo,
                    **_argumentos_comunes(configuracion),
                    **parametros,
                )
            contexto = _contexto_ejecucion(configuracion)
            if algoritmo == "RF":
                guardados = dict(st.session_state.get(clave_criterios_rf, {}))
                guardados[criterio_rf] = {
                    "resultado": resultado,
                    "firma": firma,
                    "criterio": criterio_rf,
                    "contexto": contexto,
                    "parametros": parametros,
                }
                st.session_state[clave_criterios_rf] = guardados
                errores = dict(st.session_state.get(intentos_rf, {}))
                errores.pop(criterio_rf, None)
                st.session_state[intentos_rf] = errores
            st.session_state[resultado_modelo] = resultado
            st.session_state[f"contexto_{resultado_modelo}"] = contexto
            st.session_state[f"firma_{resultado_modelo}"] = firma
            st.session_state.pop(f"error_{resultado_modelo}", None)
            st.rerun()
        except Exception as error:
            st.session_state[f"error_{resultado_modelo}"] = str(error)
            if algoritmo == "RF":
                errores = dict(st.session_state.get(intentos_rf, {}))
                errores[criterio_rf] = str(error)
                st.session_state[intentos_rf] = errores
            st.error(f"Error al entrenar {nombre_modelo}: {error}")
    if st.session_state.get(f"error_{resultado_modelo}"):
        st.error(
            f"Último error en {nombre_modelo}: {st.session_state[f'error_{resultado_modelo}']}"
        )
    if algoritmo == "RF":
        guardados = st.session_state.get(clave_criterios_rf, {})
        errores_rf = st.session_state.get(intentos_rf, {})
        for c in ("gini", "entropy", "log_loss"):
            if c in errores_rf:
                st.warning(f"Intento previo con criterion='{c}' falló: {errores_rf[c]}")
        actual_rf = guardados.get(criterio_rf)
        if actual_rf is not None:
            if actual_rf["firma"] != firma:
                st.info(
                    "Los parámetros actuales difieren del modelo entrenado visible. Vuelva a entrenar para actualizar."
                )
            _diagnostico(
                actual_rf["resultado"],
                mostrar_figura,
                prefijo=f"lab2_{algoritmo}_{criterio_rf}",
            )
            return
    resultado_previo = st.session_state.get(resultado_modelo)
    if resultado_previo is not None:
        if st.session_state.get(f"firma_{resultado_modelo}") != firma:
            st.info(
                "Los parámetros actuales difieren del modelo entrenado visible. Vuelva a entrenar para actualizar."
            )
        _diagnostico(
            resultado_previo,
            mostrar_figura,
            prefijo=f"lab2_{algoritmo}",
        )


def _render_modelos_individuales(individuales: list) -> None:
    from ai_framework.ui.views.results_view import render_panel_modelos_individuales

    filas_comparativa = []
    modelos_params = []
    for codigo, criterio, resultado, contexto, parametros in individuales:
        nombre = (
            NOMBRES_MODELOS.get(
                codigo,
                "Naive Bayes" if codigo == "NR" else codigo,
            )
            + (f" · {criterio}" if criterio else "")
        )
        fila = {
            "Algoritmo": nombre,
            "Exactitud (%)": round(100.0 * float(resultado.metricas["accuracy"]), 2),
            "Precisión Macro": round(float(resultado.metricas["precision"]["macro"]), 4),
            "Tasa de Verdaderos Macro": round(float(resultado.metricas["recall"]["macro"]), 4),
            "F1-Score Macro": round(float(resultado.metricas["f1"]["macro"]), 4),
        }
        y_true = resultado.y_true
        y_pred = resultado.y_pred
        etiquetas_lista = list(resultado.labels)
        for etiqueta in etiquetas_lista:
            reales = y_true == etiqueta
            soporte = int(reales.sum())
            col_nombre = nombre_tasa_verdaderos(etiqueta, etiquetas_lista)
            fila[col_nombre] = (
                round(100.0 * int((y_pred[reales] == etiqueta).sum()) / soporte, 2)
                if soporte
                else "Sin casos"
            )
        filas_comparativa.append(fila)
        params_dict = getattr(resultado, "parametros", {}) or (
            resultado.modelo.get_params() if hasattr(resultado, "modelo") else {}
        )
        modelos_params.append((nombre, params_dict))

    render_panel_modelos_individuales(
        pd.DataFrame(filas_comparativa),
        modelos_params,
        titulo="Modelos entrenados individualmente",
    )


def render_resultados(
    mostrar_figura: MostrarFigura, *, incluir_experimentos: bool = False
) -> None:
    """Compara métricas de prueba de modelos individuales y del benchmark experimental."""
    hay_experimento = bool(
        st.session_state.get("resultado_lab2_experimento") is not None
        or st.session_state.get("resultado_lab2_intento_experimento") is not None
    )
    clave_criterios_rf = "resultado_lab2_modelo_RF_criterios"
    individuales = []
    guardados_rf = st.session_state.get(clave_criterios_rf, {})
    if guardados_rf:
        for criterio, entrada in sorted(guardados_rf.items()):
            resultado = entrada.get("resultado")
            if resultado is not None:
                individuales.append(
                    (
                        "RF",
                        criterio,
                        resultado,
                        entrada.get("contexto", {}),
                        entrada.get("parametros", {}),
                    )
                )
    else:
        resultado_rf = st.session_state.get("resultado_lab2_modelo_RF")
        if resultado_rf is not None:
            contexto_rf = st.session_state.get(
                "contexto_lab2_modelo_RF", {}
            )
            individuales.append(
                (
                    "RF",
                    None,
                    resultado_rf,
                    contexto_rf,
                    getattr(resultado_rf, "parametros", {}),
                )
            )
    for codigo in ("KNN", "DT", "XGBoost", "AdaBoost", "NR"):
        if f"resultado_lab2_modelo_{codigo}" in st.session_state:
            resultado = st.session_state.get(
                f"resultado_lab2_modelo_{codigo}"
            )
            if resultado is not None:
                contexto = st.session_state.get(
                    f"contexto_lab2_modelo_{codigo}", {}
                )
                individuales.append(
                    (
                        codigo,
                        None,
                        resultado,
                        contexto,
                        getattr(resultado, "parametros", {}),
                    )
                )
    individuales = [
        (codigo, criterio, resultado, contexto, parametros)
        for codigo, criterio, resultado, contexto, parametros in individuales
        if resultado is not None
    ]
    if not individuales and not hay_experimento:
        st.info(
            "Todavía no hay ejecuciones individuales guardadas. Entrene un modelo para que aparezca en esta comparación."
        )
        return

    from ai_framework.ui.views.results_view import render_layout_resultados
    from ai_framework.ui.views.classification_experiments_view import (
        render_resultado_experimento,
    )

    render_layout_resultados(
        hay_individuales=bool(individuales),
        hay_comparativa=hay_experimento,
        render_individuales_fn=lambda: _render_modelos_individuales(individuales),
        render_comparativa_fn=lambda: render_resultado_experimento(mostrar_figura, solo_comparacion=True),
        mensaje_sin_datos="Todavía no hay ejecuciones individuales guardadas. Entrene un modelo para que aparezca en esta comparación.",
    )


def _diagnostico(
    resultado, mostrar_figura: MostrarFigura, prefijo: str
) -> None:
    """Presenta métricas, matriz de confusión y predicciones del modelo de clasificación."""
    st.markdown(f"#### {resultado.algoritmo}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{resultado.metricas['accuracy']:.1%}")
    c2.metric("Precisión macro", f"{resultado.metricas['precision']['macro']:.4f}")
    c3.metric("Tasa de verdaderos macro", f"{resultado.metricas['recall']['macro']:.4f}")
    c4.metric("F1 macro", f"{resultado.metricas['f1']['macro']:.4f}")
    metricas_clase = _metricas_por_clase(resultado)
    st.markdown("**Tasa de verdaderos por clase**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Clase real": etiqueta, "Tasa de verdaderos (%)": recall}
                for etiqueta, recall in metricas_clase.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    mostrar_figura(
        VisualizadorSupervisado.matriz_confusion(resultado),
        key=f"{prefijo}_confusion",
    )
    with st.expander("Valores de la matriz de confusión"):
        st.caption("Valores de la matriz de confusión")
        st.dataframe(
            resultado.matriz_confusion,
            width="stretch",
        )
    with st.expander("Métricas por clase"):
        fig_metricas = VisualizadorSupervisado.metricas_barras(resultado)
        if fig_metricas is not None:
            mostrar_figura(
                fig_metricas,
                key=f"{prefijo}_metricas_barras",
            )
        fig_dist = VisualizadorSupervisado.distribucion_estratificada(resultado)
        if fig_dist is not None:
            mostrar_figura(
                fig_dist,
                key=f"{prefijo}_distribucion_estratificada",
            )
    with st.expander("Parámetros y metadatos"):
        tab_params, tab_meta = _obtener_pestanas(
            ["⚙️ Parámetros del modelo", "📋 Metadatos de ejecución"]
        )
        with tab_params:
            render_tabla_clave_valor(
                getattr(resultado, "parametros", resultado.modelo.get_params()),
                columna_clave="Parámetro",
                columna_valor="Valor configurado",
                vacio_mensaje="No se registraron parámetros específicos para este modelo.",
            )
        with tab_meta:
            render_tabla_clave_valor(
                getattr(resultado, "metadatos", {}),
                columna_clave="Metadato",
                columna_valor="Detalle",
                vacio_mensaje="No se registraron metadatos para esta ejecución.",
            )
    predicciones = pd.DataFrame(
        {"real": resultado.y_true, "prediccion": resultado.y_pred}
    )
    predicciones["acierto"] = predicciones["real"] == predicciones["prediccion"]
    with st.expander("Predicciones de prueba"):
        st.dataframe(
            predicciones,
            width="stretch",
        )
    st.download_button(
        "Descargar predicciones (CSV)",
        predicciones.to_csv(index=True).encode("utf-8"),
        f"predicciones_{resultado.algoritmo}.csv",
        "text/csv",
        key=f"{prefijo}_predicciones",
    )
