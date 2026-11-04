"""Derived data and configuration prerequisites for supervised workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ai_framework.data.partition import comprobar_clases, dividir_indices
from ai_framework.models.supervised import Clasificacion, Regresion
from ai_framework.utils.supervised.experiments import _particiones
from ai_framework.utils.supervised.regression_config import ALGORITMOS_LAB3


@dataclass(frozen=True)
class Availability:
    """Result of checking whether a view or execution is ready for use."""

    faltantes: tuple[str, ...] = ()

    @property
    def missing(self) -> tuple[str, ...]:
        return self.faltantes

    @property
    def ready(self) -> bool:
        return not self.faltantes

    @property
    def lista(self) -> bool:
        """Indica si todos los prerrequisitos se cumplen."""
        return not self.faltantes


Disponibilidad = Availability


def _validate_predictors(base, features, train, evaluations, config) -> None:
    """Validate variance, encoding, and missing values in predictor variables."""
    data = base[features].replace([np.inf, -np.inf], np.nan)
    retained = [
        col for col in features if data.loc[train, col].nunique(dropna=True) > 1
    ]
    if not retained:
        raise ValueError("Seleccione predictores con variación en entrenamiento.")
    categorical = [
        col for col in retained if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if categorical and not config["incluir_categoricas"]:
        raise ValueError(
            f"Active Codificar categóricas o retire estas predictoras: {', '.join(categorical)}."
        )
    if not config["imputar"]:
        indices = train
        for idx in evaluations:
            indices = indices.union(idx)
        null_cols = (
            data.loc[indices, retained]
            .columns[data.loc[indices, retained].isna().any()]
            .tolist()
        )
        if null_cols:
            raise ValueError(
                f"Active imputación de predictores o corrija nulos/no finitos en: {', '.join(null_cols)}."
            )


def _classification_folds(model, base, config, options):
    """Build train, test, and validation fold index splits."""
    train, test, validation = model._indices(
        base,
        particion=config.get("particion_global"),
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=config["estratificar"],
    )
    if options is None:
        return train, test, [(train, test)]
    if options.get("estrategia", "holdout") == "cv":
        train = train.append(validation)
        folds = options.get("folds", 5)
        y = base.loc[train, model.target]
        if y.value_counts().min() < folds:
            raise ValueError(
                f"CV necesita al menos {folds} filas por clase en entrenamiento. Use holdout o aporte más ejemplos."
            )
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=config["random_state"]
        )
        fold_splits = [
            (train.take(t), train.take(v))
            for t, v in splitter.split(np.zeros(len(train)), y)
        ]
    else:
        if validation.empty:
            train, validation = dividir_indices(
                base.loc[train],
                model.target,
                test_size=options.get("validation_size") or 0.2,
                random_state=config["random_state"],
                stratify=config["estratificar"],
            )
        fold_splits = [(train, validation)]
    return train, test, fold_splits


def _regression_candidate_available(data, base, config, fold_splits, test, options) -> None:
    """Verify that at least one candidate in regression benchmark is viable."""
    errors = []
    algorithms = options.get("algoritmos", ALGORITMOS_LAB3)
    if not algorithms:
        raise ValueError("Seleccione al menos un algoritmo del experimento.")
    for algorithm in algorithms:
        features = (
            [config.get("feature_simple")] if algorithm == "RLS" else config["features"]
        )
        try:
            if None in features:
                raise ValueError("RLS necesita un predictor numérico seleccionado.")
            candidate = Regresion(
                dataframe=data, target=get_target_configuration(config), features=features
            )
            candidate._validar_predictoras(base, algorithm)
            for train, evaluation in fold_splits:
                if algorithm in ("LassoCV", "RidgeCV") and len(train) < 5:
                    raise ValueError(
                        "Se necesitan cinco filas de train para CV interna."
                    )
                _validate_predictors(base, features, train, [evaluation, test], config)
            return
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"{algorithm}: {exc}")
    raise ValueError("Ningún candidato está preparado. " + " ".join(errors))


def get_target_configuration(config: dict) -> str:
    """Get name of configured target column."""
    return config["target"]


def supervised_availability(
    datos,
    configuracion,
    familia,
    *,
    preparacion=None,
    algoritmo=None,
    parametros=None,
    experimento=None,
) -> Disponibilidad:
    """Detect deterministic data blockages and supervised configuration issues."""
    faltantes = []
    preparacion = preparacion or {}
    if (
        preparacion.get("imputar")
        or preparacion.get("escalado", "Ninguno") != "Ninguno"
    ):
        faltantes.append(
            "En Datos y preparación, desactive imputación y escalado globales y vuelva a aplicar la limpieza. Puede mantener columnas excluidas, eliminación de duplicados y filtros de filas. Las transformaciones supervisadas se aprenden solo con train."
        )
    if datos is None or datos.empty:
        return Disponibilidad(
            (*faltantes, "Aplique una fuente con filas en Fuente de datos.")
        )
    if configuracion is None:
        return Disponibilidad(
            (
                *faltantes,
                "Se necesitan una columna objetivo y al menos una predictora. Revise la fuente en Datos y preparación.",
            )
        )
    if "tarea" in configuracion and configuracion["tarea"] != familia:
        return Disponibilidad((f"Seleccione la tarea {familia} en Configuración.",))
    try:
        features = configuracion["features"]
        if familia == "regresion" and algoritmo == "RLS":
            simple = configuracion.get("feature_simple")
            if simple is None or simple not in features:
                raise ValueError(
                    "Seleccione un predictor numérico para RLS en Configuración."
                )
            features = [simple]
        clase = Regresion if familia == "regresion" else Clasificacion
        modelo = clase(
            dataframe=datos, target=configuracion["target"], features=features
        )
        base = (
            modelo._datos_regresion()
            if familia == "regresion"
            else modelo._datos_clasificacion()
        )
        if familia == "regresion":
            if algoritmo:
                modelo._validar_predictoras(base, algoritmo)
            comunes = {
                "particion": configuracion.get("particion_global"),
                "test_size": configuracion["test_size"],
                "random_state": configuracion["random_state"],
            }
            if experimento is None:
                train, test, _ = modelo._indices(base, **comunes)
                pliegues = [(train, test)]
            else:
                train, test, _, _, pliegues = _particiones(
                    modelo,
                    base,
                    **comunes,
                    estrategia=experimento.get("estrategia"),
                    validation_size=experimento.get("validation_size", 0.2),
                    folds=experimento.get("folds", 5),
                )
            if len(test) < 2:
                raise ValueError(
                    "Prueba necesita al menos dos filas para R². Aumente el tamaño de prueba o aporte más datos."
                )
        else:
            train, test, pliegues = _classification_folds(
                modelo, base, configuracion, experimento
            )
            comprobar_clases(
                base.loc[train, modelo.target], base.loc[test, modelo.target]
            )
        for entrenamiento, evaluacion in pliegues:
            if familia == "regresion":
                if len(entrenamiento) < 2:
                    raise ValueError("Entrenamiento necesita al menos dos filas.")
                modelo._validar_objetivo(base.loc[entrenamiento, modelo.target])
                if algoritmo in ("LassoCV", "RidgeCV") and len(entrenamiento) < 5:
                    raise ValueError(
                        f"{algoritmo} necesita al menos cinco filas de entrenamiento para su validación interna."
                    )
            else:
                comprobar_clases(
                    base.loc[entrenamiento, modelo.target],
                    base.loc[evaluacion, modelo.target],
                )
                if algoritmo == "KNN" and (parametros or {}).get(
                    "n_neighbors", 5
                ) > len(entrenamiento):
                    raise ValueError(
                        "Reduzca el número de vecinos: supera las filas disponibles en entrenamiento."
                    )
            if familia != "regresion" or experimento is None:
                _validate_predictors(
                    base, features, entrenamiento, [evaluacion, test], configuracion
                )
        if familia == "regresion" and experimento is not None:
            _regression_candidate_available(
                datos, base, configuracion, pliegues, test, experimento
            )
    except (ValueError, TypeError, KeyError) as exc:
        faltantes.append(
            f"Revise Configuración de {'regresión' if familia == 'regresion' else 'clasificación'}: {exc}"
        )
        if familia == "clasificacion" and "continuos" in str(exc):
            faltantes.append(
                "Para un objetivo continuo, use Configuración. No se convierte automáticamente en clases."
            )
    return Disponibilidad(tuple(faltantes))


def has_evidence(state, family, *, only_models=False, **kwargs) -> bool:
    """Indicate whether individual or experiment results exist for family."""
    solo = kwargs.get("solo_modelos", only_models)
    lab = "lab3" if family == "regresion" else "lab2"
    models = any(
        key.startswith(f"resultado_{lab}_modelo_") and state[key] is not None
        for key in state
    )
    if solo:
        return models
    return models or any(
        state.get(f"resultado_{lab}_{suffix}") is not None
        for suffix in ("experimento", "intento_experimento")
    )

# Backward compatibility aliases
_predictoras = _validate_predictors
_pliegues_clasificacion = _classification_folds
_candidato_regresion_disponible = _regression_candidate_available
configuracion_target = get_target_configuration
disponibilidad_supervisada = supervised_availability
hay_evidencia = has_evidence
