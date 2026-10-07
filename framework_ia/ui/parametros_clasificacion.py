"""Controles pequeños y explícitos para los modelos solicitados en LAB02."""

from __future__ import annotations

import streamlit as st

NOMBRES_MODELOS = {
    "KNN": "KNN · vecinos cercanos",
    "DT": "DT · árbol de decisión",
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "AdaBoost": "AdaBoost",
}
METRICAS_SELECCION = {
    "f1_macro": "F1 macro",
    "accuracy": "Accuracy",
    "precision_macro": "Precisión macro",
    "recall_macro": "Recall macro",
}


def parametros_individuales(algoritmo: str) -> dict:
    """Expone ajustes frecuentes; la configuración estándar la define el dominio."""
    modo = st.radio(
        "Configuración del modelo",
        ["Estándar", "Personalizada"],
        key=f"lab2_modo_{algoritmo}",
        persist_state="session",
        help="Estándar utiliza los valores predeterminados del algoritmo y la semilla indicada.",
    )
    if modo == "Estándar":
        return {}
    prefijo = f"lab2_param_{algoritmo}"
    c1, c2 = st.columns(2)
    if algoritmo == "KNN":
        return {
            "n_neighbors": c1.number_input(
                "Vecinos (k)", 1, 99, 5, key=f"{prefijo}_k", persist_state="session"
            ),
            "weights": c2.selectbox(
                "Pesos",
                ["uniform", "distance"],
                key=f"{prefijo}_weights",
                persist_state="session",
            ),
            "p": st.selectbox(
                "Distancia de Minkowski",
                [2, 1],
                format_func=lambda n: (
                    "Euclidiana (p=2)" if n == 2 else "Manhattan (p=1)"
                ),
                key=f"{prefijo}_p",
                persist_state="session",
            ),
        }
    if algoritmo in {"DT", "RF"}:
        profundidad = c1.number_input(
            "Profundidad máxima (0 = sin límite)",
            0,
            100,
            0,
            key=f"{prefijo}_depth",
            persist_state="session",
        )
        resultado = {
            "max_depth": profundidad or None,
            "min_samples_split": c2.number_input(
                "Mínimo de muestras para dividir",
                2,
                100,
                2,
                key=f"{prefijo}_split",
                persist_state="session",
            ),
        }
        if algoritmo == "RF":
            resultado["n_estimators"] = st.number_input(
                "Número de árboles",
                10,
                1000,
                100,
                10,
                key=f"{prefijo}_n",
                persist_state="session",
            )
        else:
            resultado["criterion"] = st.selectbox(
                "Criterio",
                ["gini", "entropy", "log_loss"],
                key=f"{prefijo}_criterion",
                persist_state="session",
            )
        return resultado
    resultado = {
        "n_estimators": c1.number_input(
            "Número de estimadores",
            10,
            1000,
            100 if algoritmo == "XGBoost" else 50,
            10,
            key=f"{prefijo}_n",
            persist_state="session",
        ),
        "learning_rate": c2.number_input(
            "Tasa de aprendizaje",
            0.01,
            2.0,
            0.3 if algoritmo == "XGBoost" else 1.0,
            0.01,
            key=f"{prefijo}_lr",
            persist_state="session",
        ),
    }
    if algoritmo == "XGBoost":
        resultado["max_depth"] = st.number_input(
            "Profundidad máxima",
            1,
            30,
            6,
            key=f"{prefijo}_depth",
            persist_state="session",
        )
    return resultado
