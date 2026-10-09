"""Controles y firmas compartidos por clasificación individual y LAB02.

Los controles conservan sus valores al navegar. Este módulo solo coordina la
interfaz; el particionado y el ajuste de transformadores pertenecen al dominio.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from ..datos.eda import EDA
from ..resultados import ResultadoParticion


def configurar_clasificacion(
    datos: pd.DataFrame, mostrar_figura: Callable | None = None
) -> dict | None:
    """Recopila configuración de target, split y preprocesamiento."""
    if len(datos.columns) < 2:
        st.warning(
            "El dataset necesita una variable objetivo y al menos una predictora."
        )
        return None
    if st.session_state.get("clasif_target") not in datos.columns:
        st.session_state.pop("clasif_target", None)
    st.markdown("### Configuración de clasificación")
    objetivo = st.selectbox(
        "Variable objetivo (target)",
        options=datos.columns.tolist(),
        index=max(0, len(datos.columns) - 1),
        persist_state="session",
        key="clasif_target",
    )
    balance, figura_balance = EDA(dataframe=datos).balance_objetivo(objetivo)
    st.markdown("#### Balance del target en el dataset activo")
    faltantes = int(datos[objetivo].isna().sum())
    st.caption(f"Valores faltantes en target: {faltantes}")
    if balance.empty:
        st.info("El target no contiene clases observadas.")
    else:
        balance_ui = balance.reset_index(names="Clase")
        st.dataframe(balance_ui, hide_index=True, width="stretch")
        if figura_balance is not None and mostrar_figura is not None:
            mostrar_figura(figura_balance)
        clases_raras = balance.loc[balance["Cantidad"] <= 2].index.tolist()
        if clases_raras:
            st.warning(
                "Clases con dos o menos filas: "
                + ", ".join(str(clase) for clase in clases_raras)
                + ". La partición estratificada puede no ser posible."
            )
    opcional_features = [columna for columna in datos.columns if columna != objetivo]
    caracteristicas_guardadas = st.session_state.get(
        "clasif_features", opcional_features
    )
    if any(columna not in opcional_features for columna in caracteristicas_guardadas):
        st.session_state["clasif_features"] = opcional_features
    caracteristicas = st.multiselect(
        "Variables predictoras",
        options=opcional_features,
        default=opcional_features,
        persist_state="session",
        key="clasif_features",
    )
    if not caracteristicas:
        st.warning("Seleccione al menos una feature antes de entrenar.")

    st.markdown("#### Partición y reproducibilidad")
    particion_global: ResultadoParticion | None = st.session_state.get(
        "particion_global"
    )
    usar_particion_global = False
    if particion_global is not None:
        usar_particion_global = st.checkbox(
            "Usar partición calculada en Datos",
            value=False,
            persist_state="session",
            key="clasif_usar_particion_global",
            help=(
                f"Train: {len(particion_global.train)} filas · "
                f"Test: {len(particion_global.test)} filas."
            ),
        )
    else:
        st.caption(
            "No hay una partición calculada en la vista Datos. Calcule una allí "
            "para reutilizarla aquí, o configure el split manualmente."
        )

    if usar_particion_global:
        test_size = particion_global.porcentaje_test
        estado_aleatorio = particion_global.semilla
        estratificar = particion_global.columna_estratificacion is not None
        resumen = (
            f"Partición global → Train {particion_global.porcentaje_train:.0%} · "
            f"Test {particion_global.porcentaje_test:.0%}"
        )
        if particion_global.porcentaje_validacion:
            resumen += f" · Validación {particion_global.porcentaje_validacion:.0%}"
        st.caption(resumen)
    else:
        test_size = (
            st.slider(
                "Tamaño de prueba (%)",
                min_value=10,
                max_value=60,
                value=25,
                step=5,
                persist_state="session",
                key="clasif_test_size",
            )
            / 100.0
        )
        estado_aleatorio = st.number_input(
            "Random state",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
            persist_state="session",
            key="clasif_random_state",
        )
        estratificar = st.checkbox(
            "Estratificar por target",
            value=True,
            persist_state="session",
            key="clasif_strat",
        )

    st.markdown("#### Preprocesamiento")
    incluir_categoricas = st.checkbox(
        "Codificar categóricas",
        value=False,
        persist_state="session",
        key="clasif_incluir_categoricas",
    )
    imputar = st.checkbox(
        "Imputar nulos", value=True, persist_state="session", key="clasif_imputar"
    )
    estandarizar = st.checkbox(
        "Escalar (solo numéricas)",
        value=True,
        persist_state="session",
        key="clasif_escalar",
    )
    return {
        "target": objetivo,
        "features": caracteristicas,
        "test_size": test_size,
        "random_state": int(estado_aleatorio),
        "estratificar": estratificar,
        "incluir_categoricas": incluir_categoricas,
        "imputar": imputar,
        "estandarizar": estandarizar,
        "usar_particion_global": usar_particion_global,
        "particion_global": particion_global if usar_particion_global else None,
    }


def firma_clasificacion(configuracion: dict, *parametros) -> tuple:
    """Firma específica para resultados supervisados."""
    particion = configuracion.get("particion_global")
    return (
        configuracion["target"],
        tuple(configuracion["features"]),
        configuracion["test_size"],
        configuracion["random_state"],
        configuracion["estratificar"],
        configuracion["incluir_categoricas"],
        configuracion["imputar"],
        configuracion["estandarizar"],
        particion.huella if particion is not None else None,
        *parametros,
    )


def configuracion_clasificacion_actual(datos: pd.DataFrame, estado) -> dict | None:
    """Build the effective configuration from saved controls without rendering widgets."""
    if len(datos.columns) < 2:
        return None
    objetivo = estado.get("clasif_target", datos.columns[-1])
    if objetivo not in datos.columns:
        objetivo = datos.columns[-1]
    features = [columna for columna in datos.columns if columna != objetivo]
    features_guardadas = estado.get("clasif_features", features)
    if any(columna not in features for columna in features_guardadas):
        features_guardadas = features
    particion = estado.get("particion_global")
    usar_particion = particion is not None and estado.get(
        "clasif_usar_particion_global", False
    )
    return {
        "target": objetivo,
        "features": features_guardadas,
        "test_size": particion.porcentaje_test
        if usar_particion
        else state_value(estado, "clasif_test_size", 25) / 100,
        "random_state": int(
            particion.semilla
            if usar_particion
            else state_value(estado, "clasif_random_state", 42)
        ),
        "estratificar": particion.columna_estratificacion is not None
        if usar_particion
        else state_value(estado, "clasif_strat", True),
        "incluir_categoricas": state_value(estado, "clasif_incluir_categoricas", False),
        "imputar": state_value(estado, "clasif_imputar", True),
        "estandarizar": state_value(estado, "clasif_escalar", True),
        "usar_particion_global": bool(usar_particion),
        "particion_global": particion if usar_particion else None,
    }


def state_value(estado, clave: str, predeterminado):
    """Read a widget value with the same default used by its renderer."""
    return estado.get(clave, predeterminado)
