"""Vistas de clasificación individual, comparación y análisis de resultados.

La UI conserva evidencia de la ejecución. Toda selección, partición y métrica
se calcula en Clasificacion, nunca al abrir la vista de resultados.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..modelos.supervisado import Clasificacion
from ..visualizacion import VisualizadorSupervisado
from .estado_clasificacion import firma_parametros, sincronizar_ejecucion
from .parametros_clasificacion import (
    METRICAS_SELECCION,
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


def preparacion_segura() -> bool:
    preparacion = st.session_state.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        st.error(
            "La preparación actual imputó o escaló antes de separar los datos. "
            "Para evitar fuga de información, restaure el dataset original en "
            "Datos y preparación. Active la imputación y el escalado en estos "
            "controles de clasificación: se ajustarán solo con entrenamiento."
        )
        return False
    st.caption(
        "Imputación, codificación y escalado se ajustan con entrenamiento. "
        "Las filas con target vacío se excluyen. Seleccione su CSV de Potabilidad "
        "o Diabetes en Fuente de datos; el CSV de ejemplo no reemplaza los datos del profesor."
    )
    return True


def _json(datos) -> str:
    return json.dumps(datos, ensure_ascii=False, indent=2, default=str)


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
        "Explore una configuración individual. Para seleccionar variantes sin usar "
        "prueba, abra Comparar configuraciones."
    )
    segura = preparacion_segura()
    parametros = parametros_individuales(algoritmo)
    st.caption("Parámetros solicitados; los efectivos se muestran después de entrenar.")
    st.json(parametros or {"configuracion": "estándar del algoritmo"}, expanded=False)
    firma = firma_parametros({"algoritmo": algoritmo, "parametros": parametros})
    estado_modelo = f"modelo_{algoritmo}"
    resultado_modelo = f"resultado_lab2_{estado_modelo}"
    sincronizar_ejecucion(st.session_state, estado_modelo, firma)
    if st.button(
        "Entrenar modelo",
        key=f"lab2_entrenar_{algoritmo}",
        type="primary",
        disabled=not configuracion["features"] or not segura,
    ):
        st.session_state.pop(resultado_modelo, None)
        try:
            with st.spinner(f"Entrenando {nombre_modelo}…"):
                resultado = _modelo(datos, configuracion).entrenar(
                    algoritmo=algoritmo,
                    **_argumentos_comunes(configuracion),
                    **parametros,
                )
            st.session_state[resultado_modelo] = resultado
            st.session_state[f"contexto_lab2_{estado_modelo}"] = _contexto_ejecucion(
                configuracion
            )
            st.success(
                "Modelo entrenado. Las métricas siguientes corresponden a prueba."
            )
        except Exception as exc:  # noqa: BLE001 - isolate model failures at the UI boundary.
            st.error(f"No fue posible entrenar {nombre_modelo}: {exc}")
    resultado = st.session_state.get(resultado_modelo)
    if resultado is None:
        st.info("Configure el modelo y pulse Entrenar modelo para ver sus resultados.")
        return
    _diagnostico(resultado, mostrar_figura, f"clasificacion_{algoritmo}")


def _configuraciones(algoritmos: list[str]) -> tuple[dict, bool]:
    from ..modelos.supervisado.clasificacion import configuraciones_lab2

    predefinidas = configuraciones_lab2()
    configuraciones = {codigo: predefinidas[codigo] for codigo in algoritmos}
    valido = True
    modo = st.selectbox(
        "Configuraciones por familia",
        ["Variantes predefinidas", "Solo estándar", "Editar variantes"],
        key="lab2_modo_variantes",
        persist_state="session",
        help="La comparación selecciona candidatos con validación y reserva prueba para la evaluación final.",
    )
    if modo == "Solo estándar":
        configuraciones = {
            codigo: [
                variante
                for variante in predefinidas[codigo]
                if variante["nombre"] == "estandar"
            ]
            for codigo in algoritmos
        }
    elif modo == "Editar variantes":
        with st.expander("Editar parámetros avanzados", expanded=False):
            st.caption(
                'Use un objeto JSON por familia: {"KNN": [{"nombre": "...", "parametros": {...}}]}. '
                "Las variantes predefinidas se conservan en las familias que deje sin editar."
            )
            texto = st.text_area(
                "Variantes JSON",
                value="",
                key="lab2_config_json",
                persist_state="session",
                height=130,
            )
        if texto.strip():
            try:
                personalizadas = json.loads(texto)
                if not isinstance(personalizadas, dict):
                    raise TypeError("El contenido debe ser un objeto JSON.")
                desconocidos = set(personalizadas) - set(NOMBRES_MODELOS)
                if desconocidos:
                    raise ValueError(
                        f"Algoritmos desconocidos: {', '.join(sorted(desconocidos))}."
                    )
                for algoritmo in algoritmos:
                    if algoritmo in personalizadas:
                        variantes = personalizadas[algoritmo]
                        if not isinstance(variantes, list) or not variantes:
                            raise ValueError(
                                f"{algoritmo} necesita al menos una configuración."
                            )
                        for variante in variantes:
                            if (
                                not isinstance(variante, dict)
                                or not isinstance(variante.get("nombre"), str)
                                or not variante["nombre"].strip()
                                or not isinstance(variante.get("parametros"), dict)
                            ):
                                raise ValueError(
                                    "Cada variante necesita nombre y parámetros válidos."
                                )
                        configuraciones[algoritmo] = variantes
            except (ValueError, TypeError) as exc:
                st.error(f"Revise las configuraciones personalizadas: {exc}")
                valido = False
    filas = [
        {
            "Algoritmo": codigo,
            "Configuración": variante["nombre"],
            "Parámetros solicitados": _json(variante["parametros"])
            if variante["parametros"]
            else "Estándar del algoritmo",
        }
        for codigo, variantes in configuraciones.items()
        for variante in variantes
    ]
    st.markdown("#### Revisión de candidatos")
    st.caption(
        "Revise las familias, variantes y parámetros antes de ejecutar; no se entrena hasta pulsar el botón."
    )
    st.dataframe(
        pd.DataFrame(filas),
        hide_index=True,
        width="stretch",
        alt="Configuraciones y parámetros que se ejecutarán para cada algoritmo",
    )
    return configuraciones, valido


def render_experimentos(datos: pd.DataFrame, configuracion: dict) -> None:
    """Compara variantes en validación y evalúa solo los ganadores en prueba."""
    st.markdown("### Comparar configuraciones")
    st.markdown("#### 1. Elija familias y selección")
    st.info(
        "Entrenamiento ajusta los modelos; validación elige la mejor variante de cada "
        "algoritmo y el ganador global. Prueba se reserva para la evaluación final."
    )
    segura = preparacion_segura()
    algoritmos = st.multiselect(
        "Algoritmos a comparar",
        list(NOMBRES_MODELOS),
        default=list(NOMBRES_MODELOS),
        format_func=NOMBRES_MODELOS.get,
        key="lab2_algoritmos",
        persist_state="session",
    )
    c1, c2 = st.columns(2)
    metrica = c1.selectbox(
        "Métrica para seleccionar",
        list(METRICAS_SELECCION),
        format_func=METRICAS_SELECCION.get,
        key="lab2_metrica",
        persist_state="session",
    )
    particion = configuracion.get("particion_global")
    validacion_externa = (
        particion is not None
        and particion.validacion is not None
        and not particion.validacion.empty
    )
    if validacion_externa:
        validation_size = particion.porcentaje_validacion
        c2.caption(
            f"Validación reutilizada: {len(particion.validacion)} filas de la partición de Datos."
        )
    else:
        validation_size = (
            c2.slider(
                "Validación sobre entrenamiento (%)",
                10,
                30,
                20,
                5,
                key="lab2_validacion",
                persist_state="session",
                help="Porcentaje del entrenamiento disponible que se reserva para validar, después de separar prueba.",
            )
            / 100
        )
        if particion is not None:
            st.caption(
                "Se separará validación del train global. El test global se conserva intacto."
            )
        else:
            st.caption(
                f"Entrenamiento: {(1 - configuracion['test_size']) * (1 - validation_size):.0%} · "
                f"Validación: {(1 - configuracion['test_size']) * validation_size:.0%} · Prueba: {configuracion['test_size']:.0%}"
            )
    if not algoritmos:
        st.warning("Seleccione al menos un algoritmo para comparar.")
    configuraciones, valido = _configuraciones(algoritmos)
    firma = firma_parametros(
        {
            "algoritmos": algoritmos,
            "configuraciones": configuraciones,
            "metrica": metrica,
            "validation_size": validation_size,
            "json": st.session_state.get("lab2_config_json", ""),
        }
    )
    sincronizar_ejecucion(st.session_state, "experimento", firma)
    cantidad = sum(len(variantes) for variantes in configuraciones.values())
    st.markdown("#### 2. Revise y ejecute explícitamente")
    if st.button(
        f"Ejecutar comparación ({cantidad} configuraciones)",
        key="lab2_ejecutar",
        type="primary",
        disabled=not (segura and valido and algoritmos and configuracion["features"]),
    ):
        st.session_state.pop("resultado_lab2_experimento", None)
        try:
            with st.spinner("Entrenando variantes y comparando en validación…"):
                resultado = _modelo(datos, configuracion).experimentar(
                    algoritmos=algoritmos,
                    configuraciones=configuraciones,
                    validation_size=validation_size,
                    metrica=metrica,
                    **_argumentos_comunes(configuracion),
                )
            st.session_state["resultado_lab2_experimento"] = {
                "resultado": resultado,
                "contexto": _contexto_ejecucion(configuracion),
                "metrica": metrica,
            }
            if resultado.mejores:
                st.success(
                    "Comparación terminada. Abra Resultados de clasificación para analizar los ganadores."
                )
            else:
                st.warning(
                    "Ninguna configuración terminó correctamente. Consulte los errores en Resultados de clasificación."
                )
        except Exception as exc:  # noqa: BLE001 - keep configuration available for recovery.
            st.error(f"No fue posible comparar los modelos: {exc}")
    if st.session_state.get("resultado_lab2_experimento") and st.button(
        "Ver resultados", key="lab2_ver_resultados"
    ):
        st.session_state["vista_activa"] = "lab2_resultados"
        st.rerun()


def render_resultados(mostrar_figura: MostrarFigura) -> None:
    """Vista de lectura; nunca ajusta ni selecciona modelos al abrirse."""
    st.markdown("### Resultados de clasificación")
    individuales = [
        (codigo, st.session_state.get(f"resultado_lab2_modelo_{codigo}"))
        for codigo in (*NOMBRES_MODELOS, "NR")
    ]
    individuales = [
        (codigo, resultado)
        for codigo, resultado in individuales
        if resultado is not None
    ]
    if individuales:
        st.markdown("#### Modelos entrenados individualmente")
        st.caption(
            "Métricas de prueba de ejecuciones exploratorias; no utilice esta tabla "
            "para seleccionar hiperparámetros o comparar contra la selección de validación."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Algoritmo": NOMBRES_MODELOS.get(
                            codigo, "Naive Bayes" if codigo == "NR" else codigo
                        ),
                        "Accuracy prueba": resultado.metricas["accuracy"],
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
                    for codigo, resultado in individuales
                ]
            ),
            hide_index=True,
            width="stretch",
            alt="Métricas de prueba y parámetros de los modelos entrenados individualmente",
        )
        for codigo, resultado in individuales:
            contexto = st.session_state.get(f"contexto_lab2_modelo_{codigo}", {})
            nombre = NOMBRES_MODELOS.get(
                codigo, "Naive Bayes" if codigo == "NR" else codigo
            )
            with st.expander(f"{nombre} · prueba", expanded=False):
                st.caption(
                    f"Dataset: {contexto.get('dataset', 'desconocido')} · "
                    f"Target: {contexto.get('target', 'desconocido')} · "
                    f"Features: {', '.join(contexto.get('features', []))}"
                )
                _diagnostico(resultado, mostrar_figura, f"clasificacion_{codigo}")
    guardado = st.session_state.get("resultado_lab2_experimento")
    if guardado is None:
        if not individuales:
            st.info(
                "Todavía no hay resultados. Entrene un modelo individual o ejecute "
                "una comparación de configuraciones. Si cambió los datos o la "
                "configuración, ejecute de nuevo."
            )
        return
    resultado = guardado["resultado"]
    contexto = guardado["contexto"]
    metrica = guardado["metrica"]
    st.markdown("#### Comparación de configuraciones · selección por validación")
    st.caption(
        f"Dataset: {contexto['dataset']} · Target: {contexto['target']} · "
        f"Semilla: {contexto['random_state']} · Métrica: {METRICAS_SELECCION[metrica]}"
    )
    errores_prueba = resultado.metadatos.get("errores_prueba", {})
    if errores_prueba:
        st.error(
            "No se pudo evaluar en prueba a uno o más ganadores. "
            "La selección de validación se conserva; revise los datos y el preprocesamiento."
        )
        for algoritmo_error, detalle in errores_prueba.items():
            st.warning(f"{algoritmo_error}: {detalle}")
    if resultado.mejor_algoritmo:
        st.success(
            f"Mejor algoritmo por {METRICAS_SELECCION[metrica]} de validación: "
            f"{NOMBRES_MODELOS.get(resultado.mejor_algoritmo, resultado.mejor_algoritmo)}. "
            "La selección no utiliza las métricas de prueba."
        )
    tabla = resultado.tabla.copy()
    tabla["parametros"] = tabla["parametros"].map(
        lambda valor: _json(valor) if isinstance(valor, dict) else str(valor)
    )
    st.dataframe(
        tabla,
        hide_index=True,
        width="stretch",
        alt="Todas las variantes comparadas: métricas de validación, parámetros, selección y errores",
    )
    fallidos = tabla[tabla["estado"] != "ok"]
    if not fallidos.empty:
        st.warning(
            f"{len(fallidos)} configuraciones no terminaron. Los errores se conservan en la tabla y el CSV."
        )
    validos = tabla[tabla["estado"] == "ok"]
    if not validos.empty:
        figura = go.Figure(
            go.Bar(
                x=validos["algoritmo"] + " · " + validos["configuracion"],
                y=validos["puntuacion_validacion"],
                text=validos["puntuacion_validacion"].round(3),
                textposition="outside",
            )
        )
        figura.update_layout(
            title=f"{METRICAS_SELECCION[metrica]} de validación por configuración",
            yaxis_title=METRICAS_SELECCION[metrica],
            yaxis_range=[0, 1.1],
            xaxis_title="Algoritmo y configuración",
            margin={"b": 100},
        )
        mostrar_figura(
            figura,
            key="lab2_ranking",
            alt="Puntuación de validación de cada variante; los valores también están en la tabla",
        )
    st.download_button(
        "Descargar comparación CSV",
        tabla.to_csv(index=False).encode("utf-8"),
        "comparacion_validacion.csv",
        "text/csv",
        key="lab2_descargar_tabla",
    )
    with st.expander("Configuración y reproducibilidad"):
        evidencia = {
            "configuracion": contexto,
            "metrica_seleccion": metrica,
            "metadatos": resultado.metadatos,
        }
        st.json(evidencia, expanded=False)
        st.download_button(
            "Descargar metadatos JSON",
            _json(evidencia),
            "metadatos_comparacion.json",
            "application/json",
            key="lab2_descargar_meta",
        )
    if not resultado.mejores:
        return
    st.markdown("### Evaluación final de los ganadores en prueba")
    resumen = []
    for algoritmo, modelo in resultado.mejores.items():
        resumen.append(
            {
                "Algoritmo": algoritmo,
                "Accuracy prueba": modelo.metricas["accuracy"],
                "F1 macro prueba": modelo.metricas["f1"]["macro"],
                "Parámetros efectivos": _json(modelo.parametros),
            }
        )
    st.dataframe(
        pd.DataFrame(resumen),
        hide_index=True,
        width="stretch",
        alt="Accuracy y F1 macro de prueba para cada ganador seleccionado en validación",
    )
    algoritmo = st.selectbox(
        "Ganador a analizar",
        list(resultado.mejores),
        format_func=NOMBRES_MODELOS.get,
        key="lab2_resultado_algoritmo",
    )
    _diagnostico(resultado.mejores[algoritmo], mostrar_figura, "lab2_ganador")


def _diagnostico(resultado, mostrar_figura: MostrarFigura, prefijo: str) -> None:
    st.markdown(f"#### {resultado.algoritmo} · resultados de prueba")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{resultado.metricas['accuracy']:.4f}")
    c2.metric("Precisión macro", f"{resultado.metricas['precision']['macro']:.4f}")
    c3.metric("Recall macro", f"{resultado.metricas['recall']['macro']:.4f}")
    c4.metric("F1 macro", f"{resultado.metricas['f1']['macro']:.4f}")
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
    st.dataframe(
        resultado.matriz_confusion,
        width="stretch",
        alt="Valores de la matriz de confusión de prueba",
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
        st.dataframe(
            predicciones,
            width="stretch",
            alt="Clases reales, predicciones y aciertos de cada fila de prueba",
        )
    st.download_button(
        "Descargar predicciones de prueba",
        predicciones.to_csv(index=True).encode("utf-8"),
        f"predicciones_{resultado.algoritmo}.csv",
        "text/csv",
        key=f"{prefijo}_predicciones",
    )
