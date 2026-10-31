"""Prerrequisitos derivados de datos y configuración para flujos supervisados."""

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
class Disponibilidad:
    """Resultado de verificar si una vista o ejecución está lista para usarse."""

    faltantes: tuple[str, ...] = ()

    @property
    def lista(self) -> bool:
        """Indica si todos los prerrequisitos se cumplen."""
        return not self.faltantes


def _predictoras(base, features, train, evaluaciones, configuracion) -> None:
    """Valida variación, codificación y nulos en las variables predictoras."""
    datos = base[features].replace([np.inf, -np.inf], np.nan)
    retenidas = [
        col for col in features if datos.loc[train, col].nunique(dropna=True) > 1
    ]
    if not retenidas:
        raise ValueError("Seleccione predictores con variación en entrenamiento.")
    categoricas = [
        col for col in retenidas if not pd.api.types.is_numeric_dtype(datos[col])
    ]
    if categoricas and not configuracion["incluir_categoricas"]:
        raise ValueError(
            f"Active Codificar categóricas o retire estas predictoras: {', '.join(categoricas)}."
        )
    if not configuracion["imputar"]:
        indices = train
        for indice in evaluaciones:
            indices = indices.union(indice)
        nulas = (
            datos.loc[indices, retenidas]
            .columns[datos.loc[indices, retenidas].isna().any()]
            .tolist()
        )
        if nulas:
            raise ValueError(
                f"Active imputación de predictores o corrija nulos/no finitos en: {', '.join(nulas)}."
            )


def _pliegues_clasificacion(modelo, base, config, opciones):
    """Construye los índices de entrenamiento, prueba y pliegues de validación."""
    train, test, validacion = modelo._indices(
        base,
        particion=config.get("particion_global"),
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=config["estratificar"],
    )
    if opciones is None:
        return train, test, [(train, test)]
    if opciones.get("estrategia", "holdout") == "cv":
        train = train.append(validacion)
        folds = opciones.get("folds", 5)
        y = base.loc[train, modelo.target]
        if y.value_counts().min() < folds:
            raise ValueError(
                f"CV necesita al menos {folds} filas por clase en entrenamiento. Use holdout o aporte más ejemplos."
            )
        divisor = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=config["random_state"]
        )
        pliegues = [
            (train.take(t), train.take(v))
            for t, v in divisor.split(np.zeros(len(train)), y)
        ]
    else:
        if validacion.empty:
            train, validacion = dividir_indices(
                base.loc[train],
                modelo.target,
                test_size=opciones.get("validation_size") or 0.2,
                random_state=config["random_state"],
                stratify=config["estratificar"],
            )
        pliegues = [(train, validacion)]
    return train, test, pliegues


def _candidato_regresion_disponible(datos, base, config, pliegues, test, opciones) -> None:
    """Verifica que al menos un candidato del benchmark de regresión sea viable."""
    errores = []
    algoritmos = opciones.get("algoritmos", ALGORITMOS_LAB3)
    if not algoritmos:
        raise ValueError("Seleccione al menos un algoritmo del experimento.")
    for algoritmo in algoritmos:
        features = (
            [config.get("feature_simple")] if algoritmo == "RLS" else config["features"]
        )
        try:
            if None in features:
                raise ValueError("RLS necesita un predictor numérico seleccionado.")
            candidato = Regresion(
                dataframe=datos, target=configuracion_target(config), features=features
            )
            candidato._validar_predictoras(base, algoritmo)
            for train, evaluacion in pliegues:
                if algoritmo in ("LassoCV", "RidgeCV") and len(train) < 5:
                    raise ValueError(
                        "Se necesitan cinco filas de train para CV interna."
                    )
                _predictoras(base, features, train, [evaluacion, test], config)
            return
        except (ValueError, TypeError, KeyError) as exc:
            errores.append(f"{algoritmo}: {exc}")
    raise ValueError("Ningún candidato está preparado. " + " ".join(errores))


def configuracion_target(config: dict) -> str:
    """Obtiene el nombre de la columna objetivo configurada."""
    return config["target"]


def disponibilidad_supervisada(
    datos,
    configuracion,
    familia,
    *,
    preparacion=None,
    algoritmo=None,
    parametros=None,
    experimento=None,
) -> Disponibilidad:
    """Detecta bloqueos deterministas de datos y configuración supervisada."""
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
            train, test, pliegues = _pliegues_clasificacion(
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
                _predictoras(
                    base, features, entrenamiento, [evaluacion, test], configuracion
                )
        if familia == "regresion" and experimento is not None:
            _candidato_regresion_disponible(
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


def hay_evidencia(estado, familia, *, solo_modelos=False) -> bool:
    """Indica si existen resultados individuales o de experimentación para la familia."""
    lab = "lab3" if familia == "regresion" else "lab2"
    modelos = any(
        clave.startswith(f"resultado_{lab}_modelo_") and estado[clave] is not None
        for clave in estado
    )
    if solo_modelos:
        return modelos
    return modelos or any(
        estado.get(f"resultado_{lab}_{sufijo}") is not None
        for sufijo in ("experimento", "intento_experimento")
    )
