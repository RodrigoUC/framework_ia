"""Catálogo de agentes y proveedores de LLM compartido por núcleo, scripts e interfaz."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionProveedorLLM:
    """Endpoint compatible con la API de chat de OpenAI, sin dependencias de Streamlit."""

    etiqueta: str
    base_url: str
    variable_clave: str | None
    modelo_sugerido: str  # los catálogos gratuitos cambian; verificar en el proveedor


@dataclass(frozen=True)
class ConfiguracionAgente:
    etiqueta: str
    descripcion: str


@dataclass(frozen=True)
class UmbralesEDA:
    """Límites de las reglas proactivas del agente de exploración."""

    nulos: float = 30.0  # % por columna
    fuga: float = 0.98  # |r| predictora-target
    desbalance: float = 0.10  # proporción de la clase minoritaria
    min_filas_identificador: int = 20
    max_clases_clasificacion: int = 10  # más valores numéricos -> regresión


@dataclass(frozen=True)
class UmbralesClustering:
    """Límites de las reglas proactivas del agente de agrupamiento."""

    silhouette_debil: float = 0.25  # por debajo no hay estructura sustancial
    cluster_pequeno: float = 0.05  # proporción mínima de filas por cluster
    nulos: float = 30.0  # % por columna, imputados antes de agrupar
    k_max: int = 10
    min_filas_identificador: int = 20


PROVEEDOR_SIN_LLM = "reglas"
PROVEEDORES_LLM = {
    "gemini": ConfiguracionProveedorLLM(
        "Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY", "gemini-2.5-flash",
    ),
    "groq": ConfiguracionProveedorLLM(
        "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile",
    ),
    "openrouter": ConfiguracionProveedorLLM(
        "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "",
    ),
    "ollama": ConfiguracionProveedorLLM(
        "Ollama (local)", "http://localhost:11434/v1", None, "llama3.2",
    ),
}
NOMBRES_PROVEEDORES = {
    PROVEEDOR_SIN_LLM: "Sin LLM (solo reglas)",
    **{codigo: proveedor.etiqueta for codigo, proveedor in PROVEEDORES_LLM.items()},
}
AGENTE_EDA = ConfiguracionAgente(
    "Agente EDA",
    "Perfila el dataset, detecta riesgos y sugiere la configuración de modelado.",
)
AGENTE_CLUSTERING = ConfiguracionAgente(
    "Agente de clustering",
    "Compara K-Means y HAC, recomienda el número de clusters y describe cada grupo.",
)
REGLAS_EXCLUSION_EDA = frozenset({"constante", "identificador", "fuga"})
