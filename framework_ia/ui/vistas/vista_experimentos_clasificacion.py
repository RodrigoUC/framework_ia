"""Vista de ejecución y evidencia experimental de clasificación."""

from __future__ import annotations

import platform
from copy import deepcopy

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from framework_ia.exportacion import (
    construir_archivos_experimento,
    construir_zip_experimento,
)
from framework_ia.ui.controladores.controlador_clasificacion import (
    _argumentos_comunes,
    _argumentos_exportacion,
    _contexto_ejecucion,
    _fuente_ejecutada,
    _json,
    _modelo,
)
from framework_ia.ui.modelos.catalogo_vistas import (
    ESTRATEGIAS,
    FOLDS_LAB2,
    TOTAL_CANDIDATOS,
)
from framework_ia.ui.modelos.disponibilidad import disponibilidad_supervisada
from framework_ia.ui.modelos.estado_sesion import firma_parametros
from framework_ia.ui.modelos.perfil_diabetes import estado_fuente_diabetes
from framework_ia.ui.vistas.navegacion import mostrar_faltantes
from framework_ia.ui.vistas.tema import MostrarFigura
from framework_ia.ui.vistas.vista_clasificacion import (
    _diagnostico,
    preparacion_segura,
)
from framework_ia.ui.vistas.vista_controles import _mostrar_dataframe
from framework_ia.utils import (
    ALGORITMOS_LAB2,
    METRICAS_SELECCION,
    configuraciones_experimento,
)
from framework_ia.visualizacion import VisualizadorSupervisado


def render_experimentos(
    datos: pd.DataFrame, configuracion: dict, mostrar_figura: MostrarFigura
) -> None:
    """Renderiza los controles y ejecuta el benchmark de clasificación."""
    st.caption(f"{TOTAL_CANDIDATOS} configuraciones · 5 algoritmos")
    contexto = _contexto_ejecucion(configuracion)
    st.caption(
        f"Dataset: {contexto['dataset']} · Objetivo: {configuracion['target']} · "
        f"Semilla: {configuracion['random_state']}"
    )
    segura = preparacion_segura(mostrar_guia=True)
    original_diabetes = (
        estado_fuente_diabetes(datos, st.session_state) == "original"
    )
    catalogo = configuraciones_experimento()
    with st.expander(
        f"Catálogo · 5 algoritmos / {TOTAL_CANDIDATOS} configuraciones"
    ):
        _mostrar_dataframe(
            pd.DataFrame(
                [
                    {
                        "Algoritmo": algoritmo,
                        "Configuración": candidato["nombre"],
                        "Parámetros solicitados": _json(candidato["parametros"])
                        if candidato["parametros"]
                        else "Estándar del estimador",
                    }
                    for algoritmo, candidatos in catalogo.items()
                    for candidato in candidatos
                ]
            ),
            alt="Catálogo de las cinco familias: estándar y variantes de cada una",
            hide_index=True,
            width="stretch",
        )
    st.markdown("#### Selección y validación")
    izquierda, derecha = st.columns(2)
    metrica = izquierda.selectbox(
        "Métrica para seleccionar",
        list(METRICAS_SELECCION),
        format_func=METRICAS_SELECCION.get,
        key="lab2_benchmark_metrica",
        persist_state="session",
    )
    estrategia = st.radio(
        "Estrategia de validación",
        list(ESTRATEGIAS),
        format_func=ESTRATEGIAS.get,
        key="lab2_benchmark_estrategia",
        horizontal=True,
        help="La validación cruzada es la usada en el informe: cada configuración se "
        "entrena 5 veces sobre train y se promedia el F1 macro.",
    )
    particion = configuracion.get("particion_global")
    validacion_externa = (
        estrategia == "holdout"
        and particion is not None
        and particion.validacion is not None
        and not particion.validacion.empty
    )
    if estrategia == "cv":
        validation_size = None
        if particion is not None:
            filas_train = len(particion.train) + (
                0
                if particion.validacion is None
                else len(particion.validacion)
            )
            st.caption(
                f"Entrenamiento: {filas_train} filas divididas en {FOLDS_LAB2} pliegues "
                f"estratificados · Prueba reservada: {len(particion.test)} filas"
            )
        else:
            st.caption(
                f"Train ({1 - configuracion['test_size']:.0%}) se divide en {FOLDS_LAB2} pliegues "
                f"estratificados; prueba ({configuracion['test_size']:.0%}) se usa una vez al final."
            )
    elif validacion_externa:
        validation_size = None
        derecha.caption(
            f"Validación reutilizada: {len(particion.validacion)} filas. "
            "Se conservan los índices de la partición compartida."
        )
        st.caption(
            f"Entrenamiento: {len(particion.train)} filas · "
            f"Validación: {len(particion.validacion)} filas · "
            f"Prueba reservada: {len(particion.test)} filas"
        )
    else:
        validation_size = (
            derecha.slider(
                "Validación sobre entrenamiento (%)",
                10,
                30,
                20,
                5,
                key="lab2_benchmark_validacion",
                persist_state="session",
                help="Fracción del train restante tras separar prueba. No es una fracción del dataset completo.",
            )
            / 100
        )
        if particion is not None:
            st.caption(
                f"Validación se extrae del train compartido ({len(particion.train)} filas). "
                f"Prueba conserva sus {len(particion.test)} filas e índices."
            )
        else:
            test_size = configuracion["test_size"]
            st.caption(
                "Proporciones previstas antes de redondear y excluir objetivos nulos: "
                f"train {(1 - test_size) * (1 - validation_size):.0%} · "
                f"validación {(1 - test_size) * validation_size:.0%} · prueba {test_size:.0%}"
            )
    opciones = {
        "metrica": metrica,
        "validation_size": validation_size,
        "estrategia": estrategia,
    }
    firma = firma_parametros(opciones)
    disponibilidad = disponibilidad_supervisada(
        datos,
        configuracion,
        "clasificacion",
        experimento={**opciones, "folds": FOLDS_LAB2},
        preparacion=st.session_state.get("dataset_preparacion"),
    )
    mostrar_faltantes(disponibilidad)
    ejecutar = st.button(
        f"Ejecutar experimentos Lab02 · {TOTAL_CANDIDATOS} configuraciones",
        key="lab2_benchmark_ejecutar",
        type="primary",
        disabled=not segura
        or not configuracion["features"]
        or original_diabetes
        or not disponibilidad.lista,
    )
    if ejecutar and segura and not original_diabetes and disponibilidad.lista:
        try:
            with st.spinner(
                f"Entrenando {TOTAL_CANDIDATOS} configuraciones y seleccionando en validación…"
            ):
                argumentos = {
                    **_argumentos_comunes(configuracion),
                    "metrica": metrica,
                }
                if estrategia == "cv":
                    argumentos.update(estrategia="cv", folds=FOLDS_LAB2)
                elif validation_size is not None:
                    argumentos["validation_size"] = validation_size
                resultado = _modelo(datos, configuracion).experimentar(
                    algoritmos=list(ALGORITMOS_LAB2),
                    configuraciones=catalogo,
                    **argumentos,
                )
            fuente = _fuente_ejecutada(datos)
            snapshot = {
                "resultado": resultado,
                "contexto": deepcopy(contexto),
                "firma": firma,
                "opciones": opciones,
                "fuente": fuente,
                "argumentos": _argumentos_exportacion(
                    configuracion, opciones, fuente
                ),
                "versiones": {
                    **resultado.metadatos.get("versiones", {}),
                    "python": platform.python_version(),
                },
            }
            if resultado.mejores:
                st.session_state["resultado_lab2_experimento"] = snapshot
                st.session_state.pop("resultado_lab2_intento_experimento", None)
            else:
                st.session_state["resultado_lab2_intento_experimento"] = (
                    snapshot
                )
            st.session_state.pop("error_lab2_experimento", None)
            st.rerun()
        except Exception as exc:
            st.session_state["error_lab2_experimento"] = str(exc)
            st.session_state.pop("resultado_lab2_intento_experimento", None)
    guardado = st.session_state.get("resultado_lab2_experimento")
    if guardado is not None and guardado["firma"] != firma:
        st.info(
            "Las opciones cambiaron. Se muestra la última ejecución guardada con "
            "su configuración original; pulse Ejecutar para sustituirla."
        )
    render_resultado_experimento(mostrar_figura, prefijo="lab2_benchmark")


def render_resultado_experimento(
    mostrar_figura: MostrarFigura,
    *,
    prefijo: str = "lab2_resultados_experimento",
    solo_comparacion: bool = False,
) -> None:
    """Renderiza el snapshot experimental de clasificación sin reentrenar modelos."""
    guardado = st.session_state.get("resultado_lab2_experimento")
    intento = st.session_state.get("resultado_lab2_intento_experimento")
    error = st.session_state.get("error_lab2_experimento")
    if error:
        st.error(f"Falló el último intento de experimentación: {error}")
        if guardado is not None:
            st.caption(
                "Se conserva la última ejecución exitosa del contexto actual."
            )
    if intento is not None:
        st.warning(
            "El último intento no produjo ganadores evaluados en prueba."
        )
        _tabla_validacion(intento["resultado"].tabla)
        for algoritmo, detalle in (
            intento["resultado"].metadatos.get("errores_prueba", {}).items()
        ):
            st.error(f"Prueba de {algoritmo}: {detalle}")
        _descargas(intento, f"{prefijo}_intento", es_intento=True)
        if guardado is not None:
            st.caption("A continuación se conserva la última ejecución exitosa.")
    if guardado is None:
        if intento is None:
            st.info(
                f"Todavía no hay experimentos guardados. Ejecute las {TOTAL_CANDIDATOS} configuraciones en Experimentos Lab02."
            )
        return
    resultado = guardado["resultado"]
    metadatos = resultado.metadatos
    metrica = guardado["opciones"]["metrica"]
    contexto = guardado["contexto"]
    st.markdown("### Experimento guardado · selección en validación")
    st.caption(
        f"Dataset: {contexto['dataset']} · Objetivo: {contexto['target']} · "
        f"Semilla: {contexto['random_state']} · Métrica ejecutada: {METRICAS_SELECCION[metrica]}"
    )
    if (
        metadatos.get("estrategia_seleccion")
        == "validacion_cruzada_estratificada"
    ):
        st.caption(
            f"Filas efectivas: train {metadatos['muestra_train']} · validación cruzada de "
            f"{metadatos['folds']} pliegues "
            f"({'/'.join(map(str, metadatos['muestra_validacion_por_pliegue']))} filas) · "
            f"prueba {metadatos['muestra_test']}"
        )
    else:
        st.caption(
            f"Filas efectivas: train {metadatos['muestra_train']} · "
            f"validación {metadatos['muestra_validacion']} · prueba {metadatos['muestra_test']}"
        )
    fallidos = metadatos.get("candidatos_fallidos", 0)
    errores_test = metadatos.get("errores_prueba", {})
    if fallidos or errores_test or len(resultado.mejores) != len(ALGORITMOS_LAB2):
        st.warning(
            f"Ejecución parcial: {fallidos} candidatos fallidos; "
            f"{len(resultado.mejores)} de 5 ganadores evaluados en prueba. "
            "Los errores forman parte de la evidencia exportada."
        )
    else:
        st.success(
            f"{TOTAL_CANDIDATOS} configuraciones evaluadas; 5 ganadores seleccionados por validación."
        )
    for algoritmo, detalle in errores_test.items():
        st.error(f"Prueba de {algoritmo}: {detalle}")
    st.markdown("#### Validación · todas las configuraciones")
    _tabla_validacion(resultado.tabla)
    validos = resultado.tabla.loc[resultado.tabla["estado"] == "ok"]
    if not validos.empty:
        figura = go.Figure(
            go.Bar(
                x=validos["puntuacion_validacion"],
                y=validos["algoritmo"] + " · " + validos["configuracion"],
                orientation="h",
                marker_color=[
                    "#1a3c2b" if elegido else "#ff8c69"
                    for elegido in validos["seleccionado"]
                ],
                text=validos["puntuacion_validacion"].map(
                    lambda valor: f"{valor:.3f}"
                ),
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Puntuación de validación: %{x:.4f}<extra></extra>",
            )
        )
        figura.update_layout(
            title=f"{METRICAS_SELECCION[metrica]} de validación",
            xaxis_title=METRICAS_SELECCION[metrica],
            xaxis_range=[0, 1],
            yaxis_title=None,
            yaxis_autorange="reversed",
            height=max(380, 32 * len(validos) + 110),
            margin={"l": 10, "r": 15, "b": 45},
        )
        mostrar_figura(
            figura,
            key=f"{prefijo}_grafico_validacion",
            alt="Puntuación de cada candidato en validación; verde identifica los ganadores por familia. Los valores están también en la tabla",
        )
    if resultado.mejor_algoritmo:
        st.markdown(
            f"**Mejor algoritmo por validación: {resultado.mejor_algoritmo}.**"
        )
    st.markdown("#### Prueba reservada · ganadores seleccionados")
    _mostrar_dataframe(
        pd.DataFrame(
            [
                {
                    "Algoritmo": algoritmo,
                    "Configuración elegida en validación": ganador.metadatos[
                        "configuracion"
                    ],
                    "Puntuación validación": ganador.metadatos[
                        "puntuacion_validacion"
                    ],
                    "Accuracy prueba": ganador.metricas["accuracy"],
                    "Precisión macro prueba": ganador.metricas["precision"][
                        "macro"
                    ],
                    "Recall macro prueba": ganador.metricas["recall"]["macro"],
                    "F1 macro prueba": ganador.metricas["f1"]["macro"],
                }
                for algoritmo, ganador in resultado.mejores.items()
            ]
        ),
        alt="Métricas de prueba de los ganadores elegidos previamente por validación, en el orden de las familias",
        hide_index=True,
        width="stretch",
    )
    if not solo_comparacion:
        with st.expander(
            "Diagnóstico del ganador · matriz de confusión y predicciones"
        ):
            algoritmos = list(resultado.mejores)
            clave_ganador = f"{prefijo}_ganador"
            if st.session_state.get(clave_ganador) not in algoritmos:
                st.session_state.pop(clave_ganador, None)
            algoritmo = st.selectbox(
                "Ganador a analizar", algoritmos, key=clave_ganador
            )
            _diagnostico(
                resultado.mejores[algoritmo],
                mostrar_figura,
                f"{prefijo}_{algoritmo}",
            )
            figura_clases = VisualizadorSupervisado.metricas_barras(
                resultado.mejores[algoritmo]
            )
            if figura_clases is not None:
                mostrar_figura(
                    figura_clases,
                    key=f"{prefijo}_{algoritmo}_clases",
                    alt="Precisión, recall y F1 por clase del ganador en prueba",
                )
    _descargas(guardado, prefijo)


def _tabla_validacion(tabla: pd.DataFrame) -> None:
    """Muestra la tabla de candidatos evaluados en validación con parámetros serializados."""
    vista = tabla.copy()
    vista["parametros"] = vista["parametros"].map(_json)
    _mostrar_dataframe(
        vista,
        alt="Configuraciones y parámetros efectivos, métricas de validación, selección, estado y errores",
        hide_index=True,
        width="stretch",
    )


def _descargas(
    guardado: dict, prefijo: str, *, es_intento: bool = False
) -> None:
    """Genera los botones de descarga del paquete ZIP, CSV y JSON del experimento."""
    st.markdown(
        "#### Descargar evidencia de este intento"
        if es_intento
        else "#### Descargar evidencia de esta ejecución"
    )
    try:
        if "archivos" not in guardado:
            archivos = construir_archivos_experimento(
                guardado["resultado"],
                argumentos=guardado["argumentos"],
                dataset=guardado["fuente"],
                versiones=guardado["versiones"],
            )
            zip_evidencia = construir_zip_experimento(archivos)
            guardado.update(archivos=archivos, zip=zip_evidencia)
        archivos = guardado["archivos"]
        st.download_button(
            "Descargar evidencia completa · ZIP",
            guardado["zip"],
            "experimentos_lab02.zip",
            "application/zip",
            key=f"{prefijo}_zip",
        )
        st.download_button(
            "Descargar comparación de validación · CSV",
            archivos["comparacion_validacion.csv"],
            "comparacion_validacion.csv",
            "text/csv",
            key=f"{prefijo}_csv",
        )
        st.download_button(
            "Descargar configuración · JSON",
            archivos["configuracion.json"],
            "configuracion.json",
            "application/json",
            key=f"{prefijo}_json",
        )
        st.caption(
            "ZIP con configuración, comparación, ganadores y matrices/predicciones de prueba. Usa el mismo formato que la CLI."
        )
        with st.expander("Configuración ejecutada y reproducibilidad"):
            st.json(
                {
                    "fuente": guardado["fuente"],
                    "experimento": guardado["resultado"].metadatos,
                },
                expanded=False,
            )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        st.error(
            f"La ejecución se conserva, pero no fue posible preparar la descarga: {exc}"
        )
