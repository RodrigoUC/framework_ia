"""Configuración explícita del CSV preparado de Diabetes para Lab02."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ..datos.diabetes_lab02 import PREDICTORAS_DIABETES
from .estado_clasificacion import limpiar_resultados_clasificacion

RUTA_PROCEDENCIA = (
    Path(__file__).resolve().parents[2] / "data" / "lab02" / "procedencia.json"
)
_CATEGORICAS = ("location", "gender", "frame")


def estado_fuente_diabetes(datos: pd.DataFrame, estado) -> str | None:
    """Reconoce sólo los bytes registrados, sin inferir perfiles por el nombre."""
    fuente = estado.get("fuente_aplicada")
    metadata = estado.get("fuente_metadata", {})
    if fuente is None or metadata.get("identidad") != fuente[2]:
        return None
    digest = metadata.get("sha256")
    if not digest:
        return None
    try:
        perfil = json.loads(RUTA_PROCEDENCIA.read_text(encoding="utf-8"))["diabetes"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(perfil, dict):
        return None
    original = perfil.get("original", {})
    derivacion = perfil.get("derivacion", {})
    if not isinstance(original, dict) or not isinstance(derivacion, dict):
        return None
    regla = derivacion.get("regla", {})
    if not isinstance(regla, dict):
        return None
    original = original.get("sha256")
    if original and digest == original:
        return "original"
    if digest != perfil.get("sha256"):
        return None
    preparacion = estado.get("dataset_preparacion", {})
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        return None
    requeridas = (*PREDICTORAS_DIABETES, "diabetes")
    if (
        datos.empty
        or not datos.index.is_unique
        or not datos.columns.is_unique
        or not set(requeridas).issubset(datos.columns)
        or not datos.equals(fuente[0])
    ):
        return None
    objetivo = datos["diabetes"]
    if objetivo.isna().any() or objetivo.nunique() != 2:
        return None
    etiquetas = (regla.get("positiva"), regla.get("negativa"))
    if (
        all(etiqueta is not None for etiqueta in etiquetas)
        and not objetivo.isin(etiquetas).all()
    ):
        return None
    numericas = [
        columna for columna in PREDICTORAS_DIABETES if columna not in _CATEGORICAS
    ]
    if any(not pd.api.types.is_numeric_dtype(datos[columna]) for columna in numericas):
        return None
    if any(pd.api.types.is_numeric_dtype(datos[columna]) for columna in _CATEGORICAS):
        return None
    return "preparado"


def _aplicar_configuracion_diabetes() -> None:
    """Callback previo a los widgets: aplica opciones sin entrenar modelos."""
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


def render_perfil_diabetes(datos: pd.DataFrame) -> None:
    """Ofrece el perfil únicamente para el CSV preparado reconocido e intacto."""
    perfil = estado_fuente_diabetes(datos, st.session_state)
    if perfil == "original":
        st.warning(
            "Este es el CSV original de Diabetes. Para los experimentos Lab02, "
            "aplique el archivo preparado data/lab02/diabetes.csv desde Fuente de datos. "
            "El preparado contiene el objetivo diabetes y conserva las variables nativas."
        )
    elif perfil == "preparado":
        st.caption(
            "Diabetes Lab02: objetivo diabetes, 17 predictoras nativas sin id ni glyhb; "
            "codificación de location, gender y frame, imputación y escalado con train."
        )
        st.button(
            "Aplicar configuración Diabetes Lab02",
            key="aplicar_configuracion_diabetes_lab02",
            on_click=_aplicar_configuracion_diabetes,
        )
        if (
            st.session_state.get("perfil_diabetes_aplicado")
            == st.session_state["fuente_aplicada"][2]
            and st.session_state.get("sup_target") == "diabetes"
            and st.session_state.get("sup_features") == list(PREDICTORAS_DIABETES)
        ):
            st.caption(
                "Configuración aplicada. Revise las opciones y ejecute los experimentos cuando esté listo."
            )
