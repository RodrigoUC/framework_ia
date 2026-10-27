"""Vistas de clasificación individual, diagnóstico y comparación de resultados."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from framework_ia.ui.controladores.controlador_clasificacion import (
    _argumentos_comunes,
    _contexto_ejecucion,
    _json,
    _metricas_por_clase,
    _metricas_porcentuales,
    _modelo,
)
from framework_ia.ui.modelos.disponibilidad import disponibilidad_supervisada
from framework_ia.ui.modelos.estado_sesion import firma_parametros
from framework_ia.ui.vistas.navegacion import mostrar_faltantes
from framework_ia.ui.vistas.tema import MostrarFigura
from framework_ia.ui.vistas.vista_controles import (
    NOMBRES_MODELOS,
    _mostrar_dataframe,
    parametros_individuales,
)
from framework_ia.visualizacion import VisualizadorSupervisado


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
            "Imputación, codificación y escalado se ajustan con entrenamiento. "
            "Las filas con target vacío se excluyen. Seleccione su CSV de Potabilidad "
            "o Diabetes en Fuente de datos; el CSV de ejemplo no reemplaza los datos del profesor."
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
                    "contexto": contexto,
                    "firma": firma,
                    "parametros": parametros,
                }
                st.session_state[clave_criterios_rf] = guardados
                st.session_state[resultado_modelo] = resultado
                st.session_state[f"contexto_lab2_{estado_modelo}"] = contexto
                errores = dict(st.session_state.get(intentos_rf, {}))
                errores.pop(criterio_rf, None)
                st.session_state[intentos_rf] = errores
            else:
                st.session_state[resultado_modelo] = resultado
                st.session_state[f"contexto_lab2_{estado_modelo}"] = contexto
            st.success(
                "Modelo entrenado. Las métricas siguientes corresponden a prueba."
            )
            st.rerun()
        except Exception as exc:
            if algoritmo == "RF":
                errores = dict(st.session_state.get(intentos_rf, {}))
                errores[criterio_rf] = str(exc)
                st.session_state[intentos_rf] = errores
            st.error(f"No fue posible entrenar {nombre_modelo}: {exc}")
    if algoritmo == "RF":
        error = st.session_state.get(intentos_rf, {}).get(criterio_rf)
        if error:
            st.error(
                f"Falló el intento actual de Random Forest ({criterio_rf}); se conserva la última ejecución exitosa: {error}"
            )
        resultado = (
            st.session_state.get(clave_criterios_rf, {})
            .get(criterio_rf, {})
            .get("resultado")
        )
    else:
        resultado = st.session_state.get(resultado_modelo)
    if resultado is None:
        st.info(
            "Configure el modelo y pulse Entrenar modelo para ver sus resultados."
        )
        return
    _diagnostico(
        resultado,
        mostrar_figura,
        f"clasificacion_{algoritmo}_{criterio_rf or 'default'}",
    )


def render_resultados(
    mostrar_figura: MostrarFigura, *, incluir_experimentos: bool = False
) -> None:
    """Vista de lectura de modelos entrenados y resultados experimentales."""
    hay_experimento = incluir_experimentos and any(
        st.session_state.get(clave)
        for clave in (
            "resultado_lab2_experimento",
            "resultado_lab2_intento_experimento",
            "error_lab2_experimento",
        )
    )
    individuales = []
    for codigo in (*NOMBRES_MODELOS, "NR"):
        if codigo == "RF":
            criterios_guardados = st.session_state.get(
                "resultado_lab2_modelo_RF_criterios", {}
            )
            if not criterios_guardados:
                legacy = st.session_state.get("resultado_lab2_modelo_RF")
                if legacy is not None:
                    parametros = getattr(legacy, "parametros", {})
                    criterio = parametros.get(
                        "criterion",
                        legacy.modelo.get_params().get("criterion", "gini"),
                    )
                    individuales.append(
                        (
                            codigo,
                            criterio,
                            legacy,
                            st.session_state.get(
                                "contexto_lab2_modelo_RF", {}
                            ),
                            parametros,
                        )
                    )
            for criterio, snapshot in criterios_guardados.items():
                individuales.append(
                    (
                        codigo,
                        criterio,
                        snapshot["resultado"],
                        snapshot["contexto"],
                        snapshot.get("parametros", {}),
                    )
                )
        else:
            resultado = st.session_state.get(f"resultado_lab2_modelo_{codigo}")
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
    if individuales:
        st.markdown("#### Modelos entrenados individualmente")
        st.caption(
            "Prueba · ejecuciones individuales. Seleccione hiperparámetros por validación, no con test."
        )
        _mostrar_dataframe(
            pd.DataFrame(
                [
                    {
                        "Algoritmo": (
                            NOMBRES_MODELOS.get(
                                codigo,
                                "Naive Bayes" if codigo == "NR" else codigo,
                            )
                            + (f" · {criterio}" if criterio else "")
                        ),
                        **_metricas_porcentuales(resultado),
                        "Precisión macro prueba": resultado.metricas[
                            "precision"
                        ]["macro"],
                        "Recall macro prueba": resultado.metricas["recall"][
                            "macro"
                        ],
                        "F1 macro prueba": resultado.metricas["f1"]["macro"],
                        "Parámetros efectivos": _json(
                            getattr(
                                resultado,
                                "parametros",
                                resultado.modelo.get_params(),
                            )
                        ),
                    }
                    for codigo, criterio, resultado, contexto, parametros in individuales
                ]
            ),
            alt="Métricas de prueba y parámetros de los modelos entrenados individualmente",
            hide_index=True,
            width="stretch",
        )
    if not individuales and not hay_experimento:
        st.info(
            "Todavía no hay ejecuciones individuales guardadas. Entrene un modelo para que aparezca en esta comparación."
        )
    if hay_experimento:
        from framework_ia.ui.vistas.vista_experimentos_clasificacion import (
            render_resultado_experimento,
        )

        render_resultado_experimento(mostrar_figura, solo_comparacion=True)


def _diagnostico(
    resultado, mostrar_figura: MostrarFigura, prefijo: str
) -> None:
    """Presenta métricas, matriz de confusión y predicciones del modelo de clasificación."""
    st.markdown(f"#### {resultado.algoritmo} · resultados de prueba")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy general", f"{resultado.metricas['accuracy']:.1%}")
    c2.metric("Precisión macro", f"{resultado.metricas['precision']['macro']:.4f}")
    c3.metric("Recall macro", f"{resultado.metricas['recall']['macro']:.4f}")
    c4.metric("F1 macro", f"{resultado.metricas['f1']['macro']:.4f}")
    metricas_clase = _metricas_por_clase(resultado)
    _mostrar_dataframe(
        pd.DataFrame(
            [
                {"Clase real": etiqueta, "Recall (%)": recall}
                for etiqueta, recall in metricas_clase.items()
            ]
        ),
        alt="Recall porcentual por clase real; los valores se calculan entre los casos reales de cada clase",
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Entrenamiento: {resultado.muestra_train} filas · Prueba: {resultado.muestra_test} filas"
    )
    st.markdown("**Matriz de confusión**")
    st.caption(
        "Filas: clase real. Columnas: clase predicha. La diagonal indica aciertos."
    )
    mostrar_figura(
        VisualizadorSupervisado.matriz_confusion(resultado),
        key=f"{prefijo}_confusion",
        alt="Matriz de confusión de prueba; filas de clases reales y columnas de predicciones",
    )

    _mostrar_dataframe(
        resultado.matriz_confusion,
        alt="Valores de la matriz de confusión de prueba",
        width="stretch",
    )
    with st.expander("Gráficos por clase y distribución estratificada"):
        fig_metricas = VisualizadorSupervisado.metricas_barras(resultado)
        if fig_metricas is not None:
            mostrar_figura(
                fig_metricas,
                key=f"{prefijo}_metricas_barras",
                alt="Comparación de precisión, recall y F1 por clase",
            )
        fig_dist = VisualizadorSupervisado.distribucion_estratificada(resultado)
        if fig_dist is not None:
            mostrar_figura(
                fig_dist,
                key=f"{prefijo}_distribucion_estratificada",
                alt="Distribución de clases en entrenamiento, prueba real y predicción",
            )
    with st.expander("Parámetros efectivos y detalles del modelo"):
        st.json(
            getattr(resultado, "parametros", resultado.modelo.get_params()),
            expanded=False,
        )
        st.json(getattr(resultado, "metadatos", {}), expanded=False)
    predicciones = pd.DataFrame(
        {"real": resultado.y_true, "prediccion": resultado.y_pred}
    )
    predicciones["acierto"] = predicciones["real"] == predicciones["prediccion"]
    with st.expander("Predicciones de prueba"):
        _mostrar_dataframe(
            predicciones,
            alt="Clases reales, predicciones y aciertos de cada fila de prueba",
            width="stretch",
        )
    st.download_button(
        "Descargar predicciones de prueba",
        predicciones.to_csv(index=True).encode("utf-8"),
        f"predicciones_{resultado.algoritmo}.csv",
        "text/csv",
        key=f"{prefijo}_predicciones",
    )
