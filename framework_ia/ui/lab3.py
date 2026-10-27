"""Entrenamiento explícito y consulta de snapshots de regresión."""

from copy import deepcopy

import pandas as pd
import streamlit as st

from ..exportacion import construir_archivos_modelo, construir_zip_experimento
from ..modelos.supervisado import Regresion
from ..utils.configuracion_regresion import ALGORITMOS_LAB3, NOMBRES_REGRESORES
from ..visualizacion import VisualizadorRegresion
from .estado_clasificacion import firma_parametros, huella_datos
from .flujo import disponibilidad_supervisada
from .guia_flujo import mostrar_faltantes
from .parametros_regresion import parametros_regresion


def argumentos_comunes(configuracion: dict) -> dict:
    return {
        **{
            clave: configuracion[clave]
            for clave in (
                "test_size",
                "random_state",
                "incluir_categoricas",
                "imputar",
                "estandarizar",
            )
        },
        "particion": configuracion.get("particion_global"),
    }


def crear_regresor(datos, configuracion, *, algoritmo=None) -> Regresion:
    features = (
        [configuracion["feature_simple"]]
        if algoritmo == "RLS"
        else configuracion["features"]
    )
    return Regresion(dataframe=datos, target=configuracion["target"], features=features)


def preparacion_regresion_segura() -> bool:
    preparacion = st.session_state.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        st.error(
            "La preparación global imputó o escaló antes del split. Desactive esas transformaciones en Datos y preparación y vuelva a aplicar sus opciones de limpieza. Configure imputación y escalado en Configuración para aprenderlos solo con train."
        )
        return False
    return True


def snapshot_ejecucion(resultado, datos, configuracion, parametros) -> dict:
    """Congela contexto y procedencia; las exportaciones no leen controles actuales."""
    argumentos = {
        clave: deepcopy(valor)
        for clave, valor in configuracion.items()
        if clave != "particion_global"
    }
    argumentos["parametros_solicitados"] = deepcopy(parametros)
    fuente = deepcopy(st.session_state.get("fuente_metadata", {}))
    fuente["preparacion_previa"] = deepcopy(
        st.session_state.get("dataset_preparacion", {})
    )
    fuente["huella_datos_efectivos"] = huella_datos(datos)
    fuente.setdefault(
        "etiqueta", st.session_state.get("fuente_aplicada", (None, "Dataset activo"))[1]
    )
    return {
        "resultado": resultado,
        "argumentos": argumentos,
        "dataset": fuente,
        "firma_parametros": firma_parametros(parametros),
    }


def mostrar_resultado_modelo(snapshot, mostrar_figura, *, clave: str) -> None:
    resultado = snapshot["resultado"]
    st.markdown(f"#### {NOMBRES_REGRESORES[resultado.algoritmo]} · ejecución guardada")
    st.caption(
        f"Target: {resultado.target} · Predictores: {', '.join(resultado.features)} · Train: {resultado.muestra_train} · Test: {resultado.muestra_test}"
    )
    columnas = st.columns(4)
    for columna, nombre in zip(columnas, ("rmse", "mae", "mse", "r2")):
        columna.metric(nombre.upper(), f"{resultado.metricas[nombre]:.4f}")
    st.caption(
        "Métricas de prueba en escala numérica, no porcentajes. R² puede ser negativo; los residuos son real menos predicción."
    )
    st.dataframe(
        pd.DataFrame({"Train": resultado.metricas_train, "Test": resultado.metricas}),
        width="stretch",
    )
    mostrar_figura(
        VisualizadorRegresion.real_predicho(resultado),
        key=f"{clave}_real",
        alt="Dispersión de valores reales y predichos en prueba; la diagonal indica predicción perfecta.",
    )
    mostrar_figura(
        VisualizadorRegresion.residuos(resultado),
        key=f"{clave}_residuos",
        alt="Residuos frente a predicción en prueba, con referencia horizontal en cero.",
    )
    with st.expander("Contexto ejecutado, parámetros y trazabilidad"):
        st.json(
            {
                "argumentos": snapshot["argumentos"],
                "dataset": snapshot["dataset"],
                "parametros_efectivos": resultado.parametros,
                "metadatos": resultado.metadatos,
            }
        )
    archivos = construir_archivos_modelo(
        resultado, argumentos=snapshot["argumentos"], dataset=snapshot["dataset"]
    )
    st.download_button(
        "Descargar evidencia del modelo (ZIP)",
        construir_zip_experimento(archivos),
        file_name=f"lab03_{resultado.algoritmo}.zip",
        mime="application/zip",
        key=f"{clave}_zip",
    )
    st.download_button(
        "Predicciones de prueba (CSV)",
        archivos["predicciones_test.csv"],
        file_name=f"{resultado.algoritmo}_test.csv",
        mime="text/csv",
        key=f"{clave}_csv",
    )


def render_modelo_regresion(
    datos, configuracion, mostrar_figura, *, algoritmo: str
) -> None:
    segura = preparacion_regresion_segura()
    parametros = parametros_regresion(algoritmo)
    disponibilidad = disponibilidad_supervisada(
        datos,
        configuracion,
        "regresion",
        algoritmo=algoritmo,
        parametros=parametros,
        preparacion=st.session_state.get("dataset_preparacion"),
    )
    mostrar_faltantes(disponibilidad)
    clave = f"resultado_lab3_modelo_{algoritmo}"
    error = f"error_lab3_modelo_{algoritmo}"
    sin_simple = algoritmo == "RLS" and configuracion["feature_simple"] is None
    if sin_simple:
        st.warning(
            "Seleccione un predictor numérico para RLS en Configuración."
        )
    if (
        st.button(
            "Entrenar modelo",
            key=f"lab3_entrenar_{algoritmo}",
            type="primary",
            disabled=not segura
            or parametros is None
            or sin_simple
            or not disponibilidad.lista,
        )
        and disponibilidad.lista
    ):
        try:
            with st.spinner("Ajustando regresión solo con entrenamiento…"):
                resultado = crear_regresor(
                    datos, configuracion, algoritmo=algoritmo
                ).entrenar(
                    algoritmo=algoritmo,
                    **argumentos_comunes(configuracion),
                    **parametros,
                )
            st.session_state[clave] = snapshot_ejecucion(
                resultado, datos, configuracion, parametros
            )
            st.session_state.pop(error, None)
            st.rerun()
        except (ValueError, TypeError, RuntimeError) as exc:
            st.session_state[error] = str(exc)
    if st.session_state.get(error):
        st.error(
            f"No se completó el intento: {st.session_state[error]}. La ejecución anterior se conserva si existe."
        )
    snapshot = st.session_state.get(clave)
    if snapshot:
        if snapshot["firma_parametros"] != firma_parametros(parametros):
            st.info(
                "Parámetros modificados sin ejecutar. Se muestra la última ejecución exitosa, no estos ajustes pendientes."
            )
        mostrar_resultado_modelo(snapshot, mostrar_figura, clave=f"lab3_{algoritmo}")
    else:
        st.info(
            "Sin resultados para este contexto. Configure el modelo y pulse Entrenar modelo."
        )


def render_resultados_regresion(mostrar_figura, *, incluir_experimentos=False) -> None:
    hay_experimento = incluir_experimentos and any(
        st.session_state.get(clave)
        for clave in (
            "resultado_lab3_experimento",
            "resultado_lab3_intento_experimento",
            "error_lab3_experimento",
        )
    )
    snapshots = {
        alg: st.session_state[f"resultado_lab3_modelo_{alg}"]
        for alg in ALGORITMOS_LAB3
        if f"resultado_lab3_modelo_{alg}" in st.session_state
    }
    if snapshots:
        st.markdown("### Modelos de regresión entrenados")
        st.caption(
            "Prueba · ejecuciones individuales. Seleccione hiperparámetros en Experimentos Lab03, no con test."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Algoritmo": alg, **snap["resultado"].metricas}
                    for alg, snap in snapshots.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    elif not hay_experimento:
        st.info("Todavía no hay modelos entrenados en este contexto.")
    if hay_experimento:
        from .experimentos_lab3 import mostrar_experimento_guardado

        mostrar_experimento_guardado(mostrar_figura, solo_comparacion=True)
