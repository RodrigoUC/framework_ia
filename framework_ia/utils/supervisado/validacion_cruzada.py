"""Selección de modelos con validación cruzada estratificada sobre train."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from ...datos.particion import comprobar_clases
from ...datos.preprocesamiento import PreprocesadorSupervisado
from ...resultados import ResultadoExperimentoClasificacion
from .configuracion_clasificacion import METRICAS_SELECCION
from .clasificacion import (
    ALGORITMOS_LAB2,
    configuraciones_lab2,
    normalizar_algoritmo,
)

COLUMNAS_CV = (
    "etapa",
    "algoritmo",
    "configuracion",
    "parametros",
    "seleccion_atributos",
    "accuracy_validacion",
    "precision_macro_validacion",
    "recall_macro_validacion",
    "f1_macro_validacion",
    "puntuacion_validacion",
    "puntuacion_validacion_std",
    "puntuaciones_pliegues",
    "seleccionado",
    "estado",
    "error",
)
REPETICIONES_PERMUTACION = 10


def _nombre_limpio(nombre: str) -> str:
    """Quita el prefijo del ColumnTransformer: numericas__ph -> ph."""
    return str(nombre).split("__", 1)[-1]


def _metricas(y_true, y_pred, labels) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def _evaluar_cv(clasificador, base, pliegues, algoritmo, random_state, opciones, parametros, labels):
    """Devuelve métricas por pliegue y los parámetros efectivos del estimador."""
    filas = []
    modelo = None
    for entrenamiento, evaluacion in pliegues:
        modelo, preparados = clasificador._ajustar(
            base, entrenamiento, algoritmo, random_state, opciones, parametros
        )
        X = preparados.transformador.transform(
            base.loc[evaluacion, clasificador.features]
        )
        filas.append(
            _metricas(base.loc[evaluacion, clasificador.target], modelo.predict(X), labels)
        )
    return pd.DataFrame(filas), modelo.get_params()


def _fila(etapa, algoritmo, nombre, parametros, seleccion):
    fila: dict[str, Any] = {c: None for c in COLUMNAS_CV}
    fila.update(
        etapa=etapa,
        algoritmo=algoritmo,
        configuracion=nombre,
        parametros=dict(parametros),
        seleccion_atributos=seleccion,
        seleccionado=False,
        estado="ok",
        error=None,
    )
    return fila


def _completar_fila(fila, tabla_pliegues, metrica, parametros_efectivos):
    medias = tabla_pliegues.mean()
    fila.update(
        {
            "accuracy_validacion": float(medias["accuracy"]),
            "precision_macro_validacion": float(medias["precision_macro"]),
            "recall_macro_validacion": float(medias["recall_macro"]),
            "f1_macro_validacion": float(medias["f1_macro"]),
            "puntuacion_validacion": float(medias[metrica]),
            "puntuacion_validacion_std": float(tabla_pliegues[metrica].std(ddof=1))
            if len(tabla_pliegues) > 1
            else 0.0,
            "puntuaciones_pliegues": [float(v) for v in tabla_pliegues[metrica]],
            "parametros": parametros_efectivos,
        }
    )


def _clase_positiva(y_train: pd.Series):
    """La clase menos frecuente de train actúa como positiva en ROC/AUC."""
    conteos = y_train.value_counts()
    return conteos.index[-1] if len(conteos) else None


def _analisis(clasificador, base, train, test, modelo, preparados, random_state, labels):
    """Diagnósticos interpretativos del ganador; no intervienen en la selección."""
    features = clasificador.features
    target = clasificador.target
    y_train = base.loc[train, target]
    y_test = base.loc[test, target]
    X_test_crudo = base.loc[test, features]
    X_test = preparados.transformador.transform(X_test_crudo)
    pred_train = modelo.predict(preparados.matriz)
    pred_test = modelo.predict(X_test)
    analisis: dict[str, Any] = {
        "metricas_train": _metricas(y_train, pred_train, labels),
        "atributos_usados": [_nombre_limpio(c) for c in preparados.matriz.columns],
    }
    positiva = _clase_positiva(y_train)
    if len(labels) == 2 and hasattr(modelo, "predict_proba"):
        try:
            probabilidades = np.asarray(modelo.predict_proba(X_test))
            clases = list(getattr(modelo, "classes_", labels))
            columna = clases.index(positiva)
            verdadero = (y_test == positiva).to_numpy()
            fpr, tpr, _ = roc_curve(verdadero, probabilidades[:, columna])
            analisis["roc"] = {
                "clase_positiva": positiva,
                "auc": float(roc_auc_score(verdadero, probabilidades[:, columna])),
                "fpr": [float(v) for v in fpr],
                "tpr": [float(v) for v in tpr],
            }
        except (ValueError, IndexError, AttributeError) as exc:
            analisis["roc"] = {"error": f"{type(exc).__name__}: {exc}"}
    referencia = f1_score(y_test, pred_test, labels=labels, average="macro", zero_division=0)
    generador = np.random.default_rng(random_state)
    permutacion = []
    for atributo in features:
        caidas = []
        for _ in range(REPETICIONES_PERMUTACION):
            alterado = X_test_crudo.copy()
            alterado[atributo] = generador.permutation(alterado[atributo].to_numpy())
            prediccion = modelo.predict(preparados.transformador.transform(alterado))
            caidas.append(
                referencia
                - f1_score(y_test, prediccion, labels=labels, average="macro", zero_division=0)
            )
        permutacion.append(
            {
                "atributo": atributo,
                "caida_f1_macro_media": float(np.mean(caidas)),
                "caida_f1_macro_std": float(np.std(caidas, ddof=1)),
            }
        )
    analisis["importancia_permutacion"] = sorted(
        permutacion, key=lambda fila: fila["caida_f1_macro_media"], reverse=True
    )
    internas = getattr(modelo, "feature_importances_", None)
    if internas is not None:
        analisis["importancia_modelo"] = sorted(
            (
                {"atributo": _nombre_limpio(nombre), "importancia": float(valor)}
                for nombre, valor in zip(preparados.matriz.columns, internas)
            ),
            key=lambda fila: fila["importancia"],
            reverse=True,
        )
    if hasattr(modelo, "tree_"):
        analisis["arbol"] = {
            "profundidad": int(modelo.get_depth()),
            "hojas": int(modelo.get_n_leaves()),
            "texto": _texto_arbol(modelo, preparados, max_depth=3),
        }
    return analisis


def _escalas_originales(preparados) -> dict[str, tuple[float, float]]:
    """Media y desviación del escalado aprendido en train para cada numérica."""
    transformador = preparados.transformador.transformador_
    numericas = transformador.named_transformers_.get("numericas")
    if numericas is None or "escalar" not in getattr(numericas, "named_steps", {}):
        return {}
    escalador = numericas.named_steps["escalar"]
    columnas = preparados.transformador.numericas_
    return {
        f"numericas__{columna}": (float(media), float(escala))
        for columna, media, escala in zip(columnas, escalador.mean_, escalador.scale_)
    }


def _texto_arbol(modelo, preparados, *, max_depth: int = 3) -> str:
    """Reglas del árbol con umbrales devueltos a las unidades originales."""
    arbol = modelo.tree_
    columnas = list(preparados.matriz.columns)
    escalas = _escalas_originales(preparados)
    clases = list(modelo.classes_)
    lineas: list[str] = []

    def condicion(nodo: int, izquierda: bool) -> str:
        columna = columnas[arbol.feature[nodo]]
        umbral = float(arbol.threshold[nodo])
        if columna.startswith("categoricas__"):
            nombre = _nombre_limpio(columna)
            return f"{nombre} {'no es' if izquierda else 'es'} verdadero"
        if columna in escalas:
            media, escala = escalas[columna]
            umbral = umbral * escala + media
        operador = "<=" if izquierda else ">"
        return f"{_nombre_limpio(columna)} {operador} {umbral:.2f}"

    def visitar(nodo: int, profundidad: int) -> None:
        sangria = "|   " * profundidad
        if arbol.children_left[nodo] == -1 or profundidad >= max_depth:
            conteos = arbol.value[nodo][0]
            total = int(arbol.n_node_samples[nodo])
            clase = clases[int(np.argmax(conteos))]
            sufijo = " (rama truncada)" if arbol.children_left[nodo] != -1 else ""
            lineas.append(f"{sangria}|--- clase: {clase} [n={total}]{sufijo}")
            return
        lineas.append(f"{sangria}|--- {condicion(nodo, True)}")
        visitar(int(arbol.children_left[nodo]), profundidad + 1)
        lineas.append(f"{sangria}|--- {condicion(nodo, False)}")
        visitar(int(arbol.children_right[nodo]), profundidad + 1)

    visitar(0, 0)
    return "\n".join(lineas) + "\n"


def _candidatos(algoritmos, configuraciones):
    from .experimentos import _candidatos_clasificacion
    return _candidatos_clasificacion(algoritmos, configuraciones)


def _parametros_registrados(efectivos: dict, solicitados: dict) -> dict:
    from .experimentos import _parametros_registrados as pr
    return pr(efectivos, solicitados)


def ejecutar_experimento_cv(
    clasificador,
    *,
    algoritmos=None,
    configuraciones=None,
    particion=None,
    test_size=0.25,
    random_state=42,
    stratify=True,
    metrica="f1_macro",
    incluir_categoricas=False,
    imputar=True,
    estandarizar=True,
    folds=5,
    seleccion_k="auto",
) -> ResultadoExperimentoClasificacion:
    """Ejecuta validación cruzada estratificada sobre train para clasificación."""
    if metrica not in METRICAS_SELECCION:
        raise ValueError(f"Métrica de selección no soportada: {metrica}.")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds debe ser un entero mayor o igual que 2.")
    catalogo = _candidatos(algoritmos, configuraciones)
    base = clasificador._datos_clasificacion()
    train, test, validacion = clasificador._indices(
        base,
        particion=particion,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    if not validacion.empty:
        train = train.append(validacion)
    target = clasificador.target
    y_train = base.loc[train, target]
    comprobar_clases(y_train, base.loc[test, target])
    if y_train.value_counts().min() < folds:
        raise ValueError(
            f"Cada clase necesita al menos {folds} filas en train para {folds} pliegues."
        )
    divisor = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    pliegues = [
        (train.take(entrenamiento), train.take(evaluacion))
        for entrenamiento, evaluacion in divisor.split(np.zeros(len(train)), y_train)
    ]
    labels = clasificador._ordenar_labels(y_train, base.loc[test, target])
    opciones = {
        "incluir_categoricas": incluir_categoricas,
        "imputar": imputar,
        "estandarizar": estandarizar,
    }
    referencia = PreprocesadorSupervisado(random_state=random_state, **opciones).fit(
        base.loc[train, clasificador.features], y_train
    )
    transformados = list(referencia.get_feature_names_out())
    if seleccion_k == "auto":
        k = max(1, math.ceil(len(transformados) / 2)) if len(transformados) > 2 else None
    else:
        k = None if seleccion_k in (None, 0) else int(seleccion_k)
    ranking = (
        PreprocesadorSupervisado(
            random_state=random_state, seleccion_k=len(transformados), **opciones
        )
        .fit(base.loc[train, clasificador.features], y_train)
        .puntajes_informacion_mutua()
    )
    metadatos = clasificador._metadatos(
        train,
        test,
        pd.Index([]),
        random_state=random_state,
        particion=particion,
        opciones=opciones,
    )
    metadatos.update(
        {
            "estrategia_seleccion": "validacion_cruzada_estratificada",
            "folds": folds,
            "metrica_seleccion": metrica,
            "validacion_externa": False,
            "reajustado_train_completo": True,
            "reajustado_train_validacion": False,
            "test_usado_para_seleccion": False,
            "desempate": "primera configuracion declarada; ante empate se conservan todos los atributos",
            "algoritmos": list(catalogo),
            "configuraciones": catalogo,
            "muestra_train": len(train),
            "muestra_validacion": None,
            "muestra_validacion_por_pliegue": [len(evaluacion) for _, evaluacion in pliegues],
            "muestra_test": len(test),
            "pliegues_indices": [evaluacion.tolist() for _, evaluacion in pliegues],
            "distribuciones": {
                nombre: base.loc[indices, target].value_counts().to_dict()
                for nombre, indices in (("train", train), ("test", test))
            },
            "seleccion_atributos": {
                "metodo": "SelectKBest(mutual_info_classif) ajustado en cada pliegue",
                "k": k,
                "atributos_transformados": [_nombre_limpio(c) for c in transformados],
                "ranking_informacion_mutua_train": {
                    _nombre_limpio(nombre): float(valor)
                    for nombre, valor in (ranking.items() if ranking is not None else [])
                },
            },
        }
    )
    try:
        metadatos["versiones"]["xgboost"] = version("xgboost")
    except PackageNotFoundError:
        metadatos["versiones"]["xgboost"] = "no instalado"

    filas: list[dict[str, Any]] = []
    ganadores: dict[str, dict[str, Any]] = {}
    for algoritmo, configs in catalogo.items():
        for config in configs:
            fila = _fila("configuracion", algoritmo, config["nombre"], config["parametros"], False)
            try:
                tabla, efectivos = _evaluar_cv(
                    clasificador, base, pliegues, algoritmo, random_state, opciones,
                    config["parametros"], labels,
                )
                _completar_fila(fila, tabla, metrica, _parametros_registrados(efectivos, config["parametros"]))
                anterior = ganadores.get(algoritmo)
                if anterior is None or fila["puntuacion_validacion"] > anterior["score"]:
                    ganadores[algoritmo] = {
                        "score": fila["puntuacion_validacion"],
                        "std": fila["puntuacion_validacion_std"],
                        "config": config,
                        "fila": len(filas),
                        "opciones": dict(opciones),
                        "seleccion": False,
                    }
            except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                fila.update(estado="error", error=f"{type(exc).__name__}: {exc}")
            filas.append(fila)

    if k is not None:
        for algoritmo, ganador in ganadores.items():
            config = ganador["config"]
            fila = _fila(
                "seleccion_atributos", algoritmo, f"{config['nombre']}+seleccion_k{k}",
                config["parametros"], True,
            )
            opciones_seleccion = {**opciones, "seleccion_k": k}
            try:
                tabla, _ = _evaluar_cv(
                    clasificador, base, pliegues, algoritmo, random_state,
                    opciones_seleccion, config["parametros"], labels,
                )
                _completar_fila(fila, tabla, metrica, filas[ganador["fila"]]["parametros"])
                if fila["puntuacion_validacion"] > ganador["score"] + 1e-12:
                    ganador.update(
                        score=fila["puntuacion_validacion"],
                        std=fila["puntuacion_validacion_std"],
                        fila=len(filas),
                        opciones=opciones_seleccion,
                        seleccion=True,
                    )
            except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                fila.update(estado="error", error=f"{type(exc).__name__}: {exc}")
            filas.append(fila)

    mejor_algoritmo = max(ganadores, key=lambda a: ganadores[a]["score"], default=None)
    metadatos["selecciones"] = {a: g["config"]["nombre"] for a, g in ganadores.items()}
    metadatos["seleccion_atributos_elegida"] = {a: g["seleccion"] for a, g in ganadores.items()}
    metadatos["mejor_algoritmo_por_validacion"] = mejor_algoritmo
    metadatos["errores_prueba"] = {}
    mejores = {}
    modelos = {}
    for algoritmo, ganador in ganadores.items():
        filas[ganador["fila"]]["seleccionado"] = True
        try:
            modelo, preparados = clasificador._ajustar(
                base, train, algoritmo, random_state, ganador["opciones"],
                ganador["config"]["parametros"],
            )
            analisis = _analisis(
                clasificador, base, train, test, modelo, preparados, random_state, labels
            )
            resultado = clasificador._resultado(
                base,
                train,
                test,
                algoritmo,
                modelo,
                preparados,
                {
                    **metadatos,
                    "conjunto_evaluacion": "test",
                    "configuracion": ganador["config"]["nombre"],
                    "seleccion_atributos_aplicada": ganador["seleccion"],
                    "puntuacion_validacion": ganador["score"],
                    "puntuacion_validacion_std": ganador["std"],
                    "analisis": analisis,
                },
            )
            mejores[algoritmo] = resultado
            modelos[algoritmo] = (modelo, preparados)
        except (ValueError, TypeError, RuntimeError) as exc:
            metadatos["errores_prueba"][algoritmo] = f"{type(exc).__name__}: {exc}"
    if mejor_algoritmo in mejores:
        clasificador.resultado = mejores[mejor_algoritmo]
        clasificador.modelo, clasificador._preparados = modelos[mejor_algoritmo]
        clasificador.exactitud = clasificador.resultado.metricas["accuracy"]
    tabla = pd.DataFrame(filas, columns=COLUMNAS_CV)
    configuracion = tabla[tabla["etapa"] == "configuracion"]
    metadatos["candidatos_exitosos"] = int((configuracion["estado"] == "ok").sum())
    metadatos["candidatos_fallidos"] = int((configuracion["estado"] == "error").sum())
    metadatos["evaluaciones_seleccion_atributos"] = int((tabla["etapa"] == "seleccion_atributos").sum())
    return ResultadoExperimentoClasificacion(
        tabla=tabla,
        mejores=mejores,
        mejor_algoritmo=mejor_algoritmo,
        metadatos=metadatos,
    )
