"""Controlador de carga de fuentes, sincronización de datos y presets."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from framework_ia.datos.diabetes import (
    PREDICTORAS_DIABETES,
    cargar_diabetes_regresion,
)
from framework_ia.datos.fuentes import CargadorCSV, ConfiguracionCSV
from framework_ia.ui.modelos.estado_clasificacion import (
    limpiar_resultados_clasificacion,
)
from framework_ia.ui.modelos.estado_regresion import (
    limpiar_resultados_regresion,
)
from framework_ia.ui.modelos.estado_sesion import firma_preparacion
from framework_ia.ui.modelos.perfil_diabetes import estado_fuente_diabetes


def _cargar_referencia_regresion() -> None:
    """Prepara la fuente local de referencia para regresión y su preset asociado."""
    modulo_app = sys.modules.get("framework_ia.ui.streamlit_app")
    cargador = getattr(modulo_app, "cargar_diabetes_regresion", cargar_diabetes_regresion)
    datos, _, metadata = cargador()
    identidad = f"builtin:sklearn-diabetes:{metadata['sha256']}"
    st.session_state["fuente_aplicada"] = (datos, metadata["archivo"], identidad)
    st.session_state["fuente_metadata"] = {**metadata, "identidad": identidad}
    st.session_state["fuente_preset_regresion"] = {
        "identidad": identidad,
        **metadata["preset_regresion"],
    }


def _aplicar_preset_regresion(identidad: str) -> None:
    """Aplica el preset de regresión una sola vez tras confirmar la fuente local."""
    preset = st.session_state.pop("fuente_preset_regresion", None)
    if preset is None or preset["identidad"] != identidad:
        return
    limpiar_resultados_regresion(st.session_state)
    st.session_state.update(
        {
            "sup_tarea": "regresion",
            "sup_target": preset["target"],
            "sup_features": list(preset["features"]),
            "sup_feature_simple": preset["feature_simple"],
            "sup_test_size": int(preset["test_size"] * 100),
            "sup_random_state": preset["random_state"],
            "sup_usar_particion_global": False,
            "sup_incluir_categoricas": False,
            "sup_imputar": True,
            "sup_escalar": True,
        }
    )


def _guardar_metadatos_fuente(
    contenido: bytes,
    configuracion: ConfiguracionCSV,
    fuente: str | None = None,
) -> None:
    """Registra los metadatos y la huella criptográfica de la fuente aplicada."""
    datos, nombre, identidad = st.session_state["fuente_aplicada"]
    st.session_state["fuente_metadata"] = {
        "archivo": nombre,
        "identidad": identidad,
        "sha256": hashlib.sha256(contenido).hexdigest(),
        "fuente": fuente.strip() if fuente and fuente.strip() else None,
        "filas": len(datos),
        "columnas": list(datos.columns),
        "lectura": {
            "separador": configuracion.separador,
            "decimal": configuracion.decimal,
            "encoding": configuracion.encoding,
            "usar_primera_columna_como_indice": configuracion.usar_primera_columna_como_indice,
        },
    }


def _cargar_fuente_csv(
    seleccion: Path,
    configuracion: ConfiguracionCSV,
    *,
    fuente: str | None = None,
) -> None:
    """Carga un archivo CSV local y conserva su identidad de archivo y lectura."""
    contenido = seleccion.read_bytes()
    datos = CargadorCSV.cargar_bytes(contenido, configuracion)
    digest = hashlib.sha256(contenido).hexdigest()
    identidad = f"local:{seleccion.resolve()}:{digest}:{configuracion}"
    st.session_state["fuente_aplicada"] = (datos, seleccion.name, identidad)
    _guardar_metadatos_fuente(contenido, configuracion, fuente)


def _limpiar_resultados() -> None:
    """Descarta resultados que ya no corresponden al dataset activo."""
    limpiar_resultados_regresion(st.session_state)
    for clave in list(st.session_state):
        if (
            clave.startswith("resultado_")
            or clave.startswith("benchmark_")
            or clave.startswith("preview_")
            or clave.startswith("comparacion_")
        ):
            del st.session_state[clave]
    st.session_state.pop("particion_global", None)
    st.session_state.pop("clasificacion_contexto", None)
    st.session_state.pop("lab2_firma_experimento", None)
    st.session_state.pop("lab2_firma_modelo", None)
    st.session_state.pop("error_lab2_experimento", None)


def _sincronizar_dataset(datos: pd.DataFrame, identidad: str) -> None:
    """Reinicia resultados únicamente cuando cambia el archivo o su lectura."""
    if st.session_state.get("dataset_identidad") == identidad:
        return
    st.session_state["dataset_identidad"] = identidad
    st.session_state["dataset_original"] = datos.copy()
    st.session_state["dataset_preparado"] = datos.copy()
    _limpiar_resultados()
    st.session_state.pop("dataset_preparacion", None)
    for clave in list(st.session_state):
        if (
            clave.startswith(
                ("modelo_", "clasif_", "reg_", "sup_", "lab3_", "dataset_")
            )
            and clave
            not in {"dataset_original", "dataset_preparado", "dataset_identidad"}
        ):
            del st.session_state[clave]


def _aplicar_configuracion_diabetes() -> None:
    """Aplica las opciones recomendadas para el CSV preparado de Diabetes."""
    datos = st.session_state.get("dataset_preparado")
    if datos is None or estado_fuente_diabetes(datos, st.session_state) != "preparado":
        return
    st.session_state.update(
        sup_tarea="clasificacion",
        sup_target="diabetes",
        sup_features=list(PREDICTORAS_DIABETES),
        sup_incluir_categoricas=True,
        sup_imputar=True,
        sup_escalar=True,
        sup_random_state=42,
        sup_test_size=25,
        sup_strat=True,
        sup_usar_particion_global=False,
        lab2_benchmark_validacion=20,
        lab2_benchmark_metrica="f1_macro",
    )
    limpiar_resultados_clasificacion(st.session_state)
    st.session_state.pop("errores_lab2_modelo_RF", None)
    st.session_state["perfil_diabetes_aplicado"] = st.session_state["fuente_aplicada"][
        2
    ]


def _usar_particion() -> None:
    """Sincroniza los controles supervisados con la partición global calculada."""
    particion = st.session_state.get("particion_global")
    if particion is not None and st.session_state.get("sup_usar_particion_global"):
        st.session_state["sup_test_size"] = round(particion.porcentaje_test * 100)
        st.session_state["sup_random_state"] = particion.semilla
        st.session_state["sup_strat"] = particion.columna_estratificacion is not None
        st.session_state["sup_particion_firma"] = firma_preparacion(st.session_state)
