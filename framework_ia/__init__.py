"""Backward compatibility layer for framework_ia -> ai_framework.

Redirects imports dynamically and exports all public symbols from ai_framework.
"""
from __future__ import annotations

import importlib
import sys
from importlib.abc import MetaPathFinder

# Map legacy submodule names to new ai_framework submodules
_LEGACY_MODULE_MAP = {
    # Data
    "framework_ia.datos": "ai_framework.data",
    "framework_ia.datos.fuentes": "ai_framework.data.sources",
    "framework_ia.datos.dataframe": "ai_framework.data.dataframe",
    "framework_ia.datos.eda": "ai_framework.data.eda",
    "framework_ia.datos.particion": "ai_framework.data.splitting",
    "framework_ia.datos.partition": "ai_framework.data.partition",
    "framework_ia.datos.preprocesamiento": "ai_framework.data.preprocessing",

    # Datasets
    "framework_ia.datasets": "ai_framework.datasets",
    "framework_ia.datasets.concrete": "ai_framework.datasets.concrete",
    "framework_ia.datasets.diabetes": "ai_framework.datasets.diabetes",
    "framework_ia.datasets.potabilidad": "ai_framework.datasets.potability",

    # Models
    "framework_ia.modelos": "ai_framework.models",
    "framework_ia.modelos.supervisado": "ai_framework.models.supervised",
    "framework_ia.modelos.supervisado.clasificacion": "ai_framework.models.supervised.classification",
    "framework_ia.modelos.supervisado.regresion": "ai_framework.models.supervised.regression",
    "framework_ia.modelos.no_supervisado": "ai_framework.models.unsupervised",
    "framework_ia.modelos.no_supervisado.cluster": "ai_framework.models.unsupervised.clustering",

    # UI
    "framework_ia.ui": "ai_framework.ui",
    "framework_ia.ui.app": "ai_framework.ui.app",
    "framework_ia.ui.streamlit_app": "ai_framework.ui.streamlit_app",
    "framework_ia.ui.i18n": "ai_framework.ui.i18n",
    "framework_ia.ui.controladores": "ai_framework.ui.controllers",
    "framework_ia.ui.controladores.controlador_clasificacion": "ai_framework.ui.controllers.classification_controller",
    "framework_ia.ui.controladores.controlador_regresion": "ai_framework.ui.controllers.regression_controller",
    "framework_ia.ui.controladores.controlador_fuentes": "ai_framework.ui.controllers.sources_controller",
    "framework_ia.ui.controladores.controlador_no_supervisado": "ai_framework.ui.controllers.unsupervised_controller",
    "framework_ia.ui.modelos": "ai_framework.ui.models",
    "framework_ia.ui.modelos.catalogo_vistas": "ai_framework.ui.models.view_catalog",
    "framework_ia.ui.modelos.disponibilidad": "ai_framework.ui.models.availability",
    "framework_ia.ui.modelos.estado_clasificacion": "ai_framework.ui.models.classification_state",
    "framework_ia.ui.modelos.estado_regresion": "ai_framework.ui.models.regression_state",
    "framework_ia.ui.modelos.estado_sesion": "ai_framework.ui.models.session_state",
    "framework_ia.ui.modelos.perfil_diabetes": "ai_framework.ui.models.diabetes_profile",
    "framework_ia.ui.vistas": "ai_framework.ui.views",
    "framework_ia.ui.vistas.navegacion": "ai_framework.ui.views.navigation",
    "framework_ia.ui.vistas.tema": "ai_framework.ui.views.theme",
    "framework_ia.ui.vistas.vista_clasificacion": "ai_framework.ui.views.classification_view",
    "framework_ia.ui.vistas.vista_configuracion": "ai_framework.ui.views.configuration_view",
    "framework_ia.ui.vistas.vista_controles": "ai_framework.ui.views.controls_view",
    "framework_ia.ui.vistas.vista_datos": "ai_framework.ui.views.data_view",
    "framework_ia.ui.vistas.vista_experimentos_clasificacion": "ai_framework.ui.views.classification_experiments_view",
    "framework_ia.ui.vistas.vista_experimentos_regresion": "ai_framework.ui.views.regression_experiments_view",
    "framework_ia.ui.vistas.vista_no_supervisado": "ai_framework.ui.views.unsupervised_view",
    "framework_ia.ui.vistas.vista_regresion": "ai_framework.ui.views.regression_view",
    "framework_ia.ui.vistas.vista_resultados": "ai_framework.ui.views.results_view",

    # Utils
    "framework_ia.utils": "ai_framework.utils",
    "framework_ia.utils.colores": "ai_framework.utils.colors",
    "framework_ia.utils.compatibilidad_sklearn": "ai_framework.utils.sklearn_compatibility",
    "framework_ia.utils.resultados": "ai_framework.utils.results",
    "framework_ia.utils.visualizacion": "ai_framework.utils.visualization",
    "framework_ia.utils.datos": "ai_framework.utils.data",
    "framework_ia.utils.datos.configuracion_csv": "ai_framework.utils.data.csv_config",
    "framework_ia.utils.datos.configuracion_particion": "ai_framework.utils.data.partition_config",
    "framework_ia.utils.datos.configuracion_preprocesamiento": "ai_framework.utils.data.preprocessing_config",
    "framework_ia.utils.supervisado": "ai_framework.utils.supervised",
    "framework_ia.utils.supervisado.algoritmos_regresion": "ai_framework.utils.supervised.regression_algorithms",
    "framework_ia.utils.supervisado.clasificacion": "ai_framework.utils.supervised.classification",
    "framework_ia.utils.supervisado.comparacion": "ai_framework.utils.supervised.comparison",
    "framework_ia.utils.supervisado.configuracion_clasificacion": "ai_framework.utils.supervised.classification_config",
    "framework_ia.utils.supervisado.configuracion_regresion": "ai_framework.utils.supervised.regression_config",
    "framework_ia.utils.supervisado.evidencia": "ai_framework.utils.supervised.evidence",
    "framework_ia.utils.supervisado.experimentos": "ai_framework.utils.supervised.experiments",
    "framework_ia.utils.supervisado.exportacion": "ai_framework.utils.supervised.export",
    "framework_ia.utils.supervisado.regresion": "ai_framework.utils.supervised.regression",
    "framework_ia.utils.supervisado.validacion_cruzada": "ai_framework.utils.supervised.cross_validation",
    "framework_ia.utils.no_supervisado": "ai_framework.utils.unsupervised",
    "framework_ia.utils.no_supervisado.comparacion": "ai_framework.utils.unsupervised.comparison",
    "framework_ia.utils.no_supervisado.evaluacion_clusters": "ai_framework.utils.unsupervised.cluster_evaluation",
    "framework_ia.utils.no_supervisado.k_medoids": "ai_framework.utils.unsupervised.k_medoids",

    # Short historical aliases
    "framework_ia.resultados": "ai_framework.utils.results",
    "framework_ia.visualizacion": "ai_framework.utils.visualization",
    "framework_ia.exportacion": "ai_framework.utils.supervised.export",
    "framework_ia.evidencia": "ai_framework.utils.supervised.evidence",
}


from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec


class _AliasLoader(Loader):
    def __init__(self, mod):
        self._mod = mod

    def create_module(self, spec):
        return self._mod

    def exec_module(self, module):
        pass


class _LegacyRedirector(MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        target_name = _LEGACY_MODULE_MAP.get(fullname)
        if not target_name and fullname.startswith("framework_ia."):
            possible = "ai_framework." + fullname[len("framework_ia."):]
            try:
                importlib.import_module(possible)
                target_name = possible
            except ModuleNotFoundError:
                pass
        if target_name:
            target_mod = importlib.import_module(target_name)
            sys.modules[fullname] = target_mod
            spec = ModuleSpec(fullname, _AliasLoader(target_mod))
            if hasattr(target_mod, "__path__"):
                spec.submodule_search_locations = list(target_mod.__path__)
            return spec
        return None


if not any(isinstance(finder, _LegacyRedirector) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyRedirector())

from ai_framework import (
    CargadorCSV,
    Clasificacion,
    Cluster,
    ConfiguracionCSV,
    ConfiguracionParticion,
    ConfiguracionPreprocesamiento,
    DataFrame,
    DatosPreparados,
    EDA,
    NoSupervisado,
    Particionador,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
    Regresion,
    ResultadoACP,
    ResultadoClasificacion,
    ResultadoCluster,
    ResultadoExperimentoClasificacion,
    ResultadoExperimentoRegresion,
    ResultadoParticion,
    ResultadoProyeccion,
    ResultadoRegresion,
    Supervisado,
)

datos = importlib.import_module("ai_framework.data")
datasets = importlib.import_module("ai_framework.datasets")
modelos = importlib.import_module("ai_framework.models")
ui = importlib.import_module("ai_framework.ui")
utils = importlib.import_module("ai_framework.utils")

resultados = importlib.import_module("ai_framework.utils.results")
visualizacion = importlib.import_module("ai_framework.utils.visualization")
exportacion = importlib.import_module("ai_framework.utils.supervised.export")
evidencia = importlib.import_module("ai_framework.utils.supervised.evidence")

sys.modules.setdefault("framework_ia.datos", datos)
sys.modules.setdefault("framework_ia.datasets", datasets)
sys.modules.setdefault("framework_ia.modelos", modelos)
sys.modules.setdefault("framework_ia.ui", ui)
sys.modules.setdefault("framework_ia.utils", utils)
sys.modules.setdefault("framework_ia.resultados", resultados)
sys.modules.setdefault("framework_ia.visualizacion", visualizacion)
sys.modules.setdefault("framework_ia.exportacion", exportacion)
sys.modules.setdefault("framework_ia.evidencia", evidencia)

__all__ = [
    "CargadorCSV",
    "Clasificacion",
    "Cluster",
    "ConfiguracionCSV",
    "ConfiguracionParticion",
    "ConfiguracionPreprocesamiento",
    "DataFrame",
    "DatosPreparados",
    "EDA",
    "NoSupervisado",
    "Particionador",
    "PreprocesadorNoSupervisado",
    "PreprocesadorSupervisado",
    "Regresion",
    "ResultadoACP",
    "ResultadoClasificacion",
    "ResultadoCluster",
    "ResultadoExperimentoClasificacion",
    "ResultadoExperimentoRegresion",
    "ResultadoParticion",
    "ResultadoProyeccion",
    "ResultadoRegresion",
    "Supervisado",
    "datos",
    "datasets",
    "modelos",
    "ui",
    "utils",
    "resultados",
    "visualizacion",
    "exportacion",
    "evidencia",
]
