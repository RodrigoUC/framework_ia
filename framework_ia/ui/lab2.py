"""Vistas de clasificación individual, comparación y análisis de resultados.

La UI conserva evidencia de la ejecución. Toda selección, partición y métrica
se calcula en Clasificacion, nunca al abrir la vista de resultados.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import streamlit as st

from ..modelos.supervisado import Clasificacion
from ..visualizacion import VisualizadorSupervisado
from .estado_clasificacion import firma_parametros
from .parametros_clasificacion import (
    NOMBRES_MODELOS,
    parametros_individuales,
)

MostrarFigura = Callable[..., None]


def _argumentos_comunes(configuracion: dict) -> dict:
    return {
        "test_size": configuracion["test_size"],
        "random_state": configuracion["random_state"],
        "stratify": configuracion["estratificar"],
        "incluir_categoricas": configuracion["incluir_categoricas"],
        "imputar": configuracion["imputar"],
        "estandarizar": configuracion["estandarizar"],
        "particion": configuracion.get("particion_global"),
    }


def _modelo(datos: pd.DataFrame, configuracion: dict) -> Clasificacion:
    return Clasificacion(
        dataframe=datos,
        target=configuracion["target"],
        features=configuracion["features"],
    )


def _contexto_ejecucion(configuracion: dict) -> dict:
    return {
        "dataset": st.session_state.get("fuente_aplicada", (None, "", ""))[1],
        **{
            clave: valor
            for clave, valor in configuracion.items()
            if clave != "particion_global"
        },
    }


def preparacion_segura(*, mostrar_guia: bool = False) -> bool:
    preparacion = st.session_state.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        st.error(
            "La preparación actual imputó o escaló antes de separar los datos. "
            "Para evitar fuga de información, restaure el dataset original en "
            "Datos y preparación. Ajuste imputación y escalado en Configuración "
            "de clasificación para que se aprendan solo con entrenamiento."
        )
        return False
    if mostrar_guia:
        st.caption(
            "Imputación, codificación y escalado se ajustan con entrenamiento. "
            "Las filas con target vacío se excluyen. Seleccione su CSV de Potabilidad "
            "o Diabetes en Fuente de datos; el CSV de ejemplo no reemplaza los datos del profesor."
        )
    return True


def _json(datos) -> str:
    return json.dumps(datos, ensure_ascii=False, indent=2, default=str)


def _mostrar_dataframe(data: pd.DataFrame, *, alt: str, **opciones) -> None:
    """Show an accessible text description beside the table across Streamlit APIs."""
    st.caption(alt)
    st.dataframe(data, **opciones)


def _metricas_por_clase(resultado) -> dict[str, float | str]:
    """Return actual-class recall percentages using saved truth/predictions."""
    y_true = resultado.y_true
    y_pred = resultado.y_pred
    etiquetas = list(resultado.labels)
    binarias_yn = (
        len(etiquetas) == 2
        and {str(etiqueta).casefold() for etiqueta in etiquetas} == {"y", "n"}
        and all(isinstance(etiqueta, str) for etiqueta in etiquetas)
    )
    recalls = {}
    for etiqueta in etiquetas:
        nombre = (
            f"Recall {str(etiqueta).upper()} (%)"
            if binarias_yn
            else f"Recall actual {etiqueta!s} (%)"
        )
        reales = y_true == etiqueta
        soporte = int(reales.sum())
        recalls[nombre] = (
            100.0 * int((y_pred[reales] == etiqueta).sum()) / soporte
            if soporte
            else "Sin casos evaluados"
        )
    return recalls


def _metricas_porcentuales(resultado) -> dict[str, float | str]:
    return {
        "Accuracy general (%)": 100.0 * float(resultado.metricas["accuracy"]),
        **_metricas_por_clase(resultado),
    }


def render_modelo_individual(
    datos: pd.DataFrame,
    configuracion: dict,
    mostrar_figura: MostrarFigura,
    *,
    algoritmo: str,
) -> None:
    """Entrenamiento exploratorio de un único modelo con parámetros visibles."""
    nombre_modelo = NOMBRES_MODELOS.get(algoritmo, "Naive Bayes")
    st.markdown(f"### {nombre_modelo}")
    st.caption(
        "Entrene una configuración; la comparación conserva esta ejecución hasta que cambie el dataset o la partición compartida."
    )
    segura = preparacion_segura()
    parametros = parametros_individuales(algoritmo)
    st.caption("Parámetros solicitados; los efectivos se muestran después de entrenar.")
    st.json(parametros or {"configuracion": "estándar del algoritmo"}, expanded=False)
    firma = firma_parametros({"algoritmo": algoritmo, "parametros": parametros})
    estado_modelo = f"modelo_{algoritmo}"
    resultado_modelo = f"resultado_lab2_{estado_modelo}"
    criterio_rf = parametros.get("criterion", "gini") if algoritmo == "RF" else None
    clave_criterios_rf = "resultado_lab2_modelo_RF_criterios"
    intentos_rf = "errores_lab2_modelo_RF"
    if st.button(
        "Entrenar modelo",
        key=f"lab2_entrenar_{algoritmo}",
        type="primary",
        disabled=not configuracion["features"] or not segura,
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
        except Exception as exc:  # noqa: BLE001 - isolate model failures at the UI boundary.
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
        st.info("Configure el modelo y pulse Entrenar modelo para ver sus resultados.")
        return
    _diagnostico(
        resultado,
        mostrar_figura,
        f"clasificacion_{algoritmo}_{criterio_rf or 'default'}",
    )


def render_resultados(mostrar_figura: MostrarFigura) -> None:
    """Vista de lectura; nunca ajusta ni selecciona modelos al abrirse."""
    st.markdown("### Resultados de clasificación")
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
                        "criterion", legacy.modelo.get_params().get("criterion", "gini")
                    )
                    individuales.append(
                        (
                            codigo,
                            criterio,
                            legacy,
                            st.session_state.get("contexto_lab2_modelo_RF", {}),
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
                contexto = st.session_state.get(f"contexto_lab2_modelo_{codigo}", {})
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
            "Métricas de prueba de ejecuciones exploratorias; no utilice esta tabla "
            "para seleccionar hiperparámetros o comparar contra la selección de validación."
        )
        _mostrar_dataframe(
            pd.DataFrame(
                [
                    {
                        "Algoritmo": (
                            NOMBRES_MODELOS.get(
                                codigo, "Naive Bayes" if codigo == "NR" else codigo
                            )
                            + (f" · {criterio}" if criterio else "")
                        ),
                        **_metricas_porcentuales(resultado),
                        "Precisión macro prueba": resultado.metricas["precision"][
                            "macro"
                        ],
                        "Recall macro prueba": resultado.metricas["recall"]["macro"],
                        "F1 macro prueba": resultado.metricas["f1"]["macro"],
                        "Parámetros efectivos": _json(
                            getattr(
                                resultado, "parametros", resultado.modelo.get_params()
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
        for codigo, criterio, resultado, contexto, parametros in individuales:
            nombre = NOMBRES_MODELOS.get(
                codigo, "Naive Bayes" if codigo == "NR" else codigo
            )
            with st.expander(
                f"{nombre}{f' · {criterio}' if criterio else ''} · prueba",
                expanded=False,
            ):
                st.caption(
                    f"Dataset: {contexto.get('dataset', 'desconocido')} · "
                    f"Target: {contexto.get('target', 'desconocido')} · "
                    f"Features: {', '.join(contexto.get('features', []))}"
                )
                st.caption(
                    f"Criterio y parámetros guardados: {criterio or 'predeterminados'} · {_json(parametros or getattr(resultado, 'parametros', {}))}"
                )
                _diagnostico(
                    resultado,
                    mostrar_figura,
                    f"clasificacion_{codigo}_{criterio or 'default'}",
                )
    if not individuales:
        st.info(
            "Todavía no hay resultados guardados. Entrene un modelo para que aparezca en esta comparación."
        )
        return


def _diagnostico(resultado, mostrar_figura: MostrarFigura, prefijo: str) -> None:
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
    # Numeric table is a keyboard and screen-reader alternative to the heatmap.
    _mostrar_dataframe(
        resultado.matriz_confusion,
        alt="Valores de la matriz de confusión de prueba",
        width="stretch",
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
