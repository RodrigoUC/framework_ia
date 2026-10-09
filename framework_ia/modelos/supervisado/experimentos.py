"""Selección de modelos mediante holdout común y prueba reservada.

El score de prueba jamás participa en la elección de configuración ni algoritmo.
Los ganadores conservan el ajuste realizado en train (sin reajuste con validación)
para que sus métricas y transformaciones sean directamente comparables.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd

from ...resultados import ResultadoExperimentoClasificacion
from ...utils import METRICAS_SELECCION as ETIQUETAS_METRICAS
from .clasificacion import ALGORITMOS_LAB2, configuraciones_lab2, normalizar_algoritmo
from .particiones import comprobar_clases, dividir_indices

METRICAS_SELECCION = tuple(ETIQUETAS_METRICAS)
COLUMNAS_TABLA = (
    "algoritmo",
    "configuracion",
    "parametros",
    "accuracy_validacion",
    "precision_macro_validacion",
    "recall_macro_validacion",
    "f1_macro_validacion",
    "puntuacion_validacion",
    "seleccionado",
    "estado",
    "error",
)


def _candidatos(algoritmos, configuraciones):
    if isinstance(algoritmos, str):
        algoritmos = [algoritmos]
    elegidos = list(
        dict.fromkeys(
            normalizar_algoritmo(a)
            for a in (ALGORITMOS_LAB2 if algoritmos is None else algoritmos)
        )
    )
    if not elegidos:
        raise ValueError("Seleccione al menos un algoritmo para experimentar.")
    catalogo = configuraciones_lab2()
    catalogo["NR"] = [{"nombre": "estandar", "parametros": {}}]
    if configuraciones is not None:
        if not isinstance(configuraciones, dict):
            raise TypeError("configuraciones debe ser un diccionario por algoritmo.")
        claves = set()
        for clave, lista in configuraciones.items():
            algoritmo = normalizar_algoritmo(clave)
            if algoritmo in claves:
                raise ValueError(
                    f"Configuraciones repetidas para {algoritmo} mediante aliases."
                )
            claves.add(algoritmo)
            catalogo[algoritmo] = lista
    resultado = {}
    for algoritmo in elegidos:
        lista = catalogo[algoritmo]
        if not isinstance(lista, (list, tuple)) or not lista:
            raise ValueError(f"{algoritmo} necesita al menos una configuración.")
        nombres = set()
        resultado[algoritmo] = []
        for config in lista:
            if not isinstance(config, dict) or set(config) != {"nombre", "parametros"}:
                raise ValueError(
                    "Cada configuración necesita exactamente nombre y parametros."
                )
            nombre, parametros = config["nombre"], config["parametros"]
            if not isinstance(nombre, str) or not nombre.strip() or nombre in nombres:
                raise ValueError(
                    f"Nombres de configuración vacíos o repetidos para {algoritmo}."
                )
            if not isinstance(parametros, dict) or not all(
                isinstance(c, str) for c in parametros
            ):
                raise ValueError(
                    "parametros debe ser un diccionario de argumentos del estimador."
                )
            # Validation is the sole selection set; never allow injected early-stop data.
            if {"eval_set", "callbacks", "early_stopping_rounds"} & set(parametros):
                raise ValueError(
                    "El benchmark no admite eval_set, callbacks ni early_stopping_rounds externos."
                )
            nombres.add(nombre)
            resultado[algoritmo].append(
                {"nombre": nombre, "parametros": dict(parametros)}
            )
    return resultado


def ejecutar_experimento(
    clasificador,
    *,
    algoritmos=None,
    configuraciones=None,
    particion=None,
    test_size=0.25,
    validation_size=0.2,
    random_state=42,
    stratify=True,
    metrica="f1_macro",
    incluir_categoricas=False,
    imputar=True,
    estandarizar=True,
) -> ResultadoExperimentoClasificacion:
    """Ejecuta un benchmark con particiones idénticas para todos los candidatos.

    ``validation_size`` es la fracción de train reservada para validar cuando la
    partición no incluye validación. ``test_size`` solo aplica sin partición.
    Los errores de un candidato quedan registrados y no abortan los restantes.
    Los errores de datos/partición sí abortan antes de entrenar cualquier modelo.
    """
    if metrica not in METRICAS_SELECCION:
        raise ValueError(
            f"Métrica de selección no soportada: {metrica}. Use {METRICAS_SELECCION}."
        )
    catalogo = _candidatos(algoritmos, configuraciones)
    base = clasificador._datos_clasificacion()
    train, test, validacion = clasificador._indices(
        base,
        particion=particion,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    validacion_externa = not validacion.empty
    if not validacion_externa:
        if not 0 < validation_size < 1:
            raise ValueError("validation_size debe estar entre 0 y 1 (excluido).")
        train, validacion = dividir_indices(
            base.loc[train],
            clasificador.target,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )
    comprobar_clases(
        base.loc[train, clasificador.target],
        base.loc[validacion, clasificador.target],
        base.loc[test, clasificador.target],
    )
    opciones = {
        "incluir_categoricas": incluir_categoricas,
        "imputar": imputar,
        "estandarizar": estandarizar,
    }
    metadatos = clasificador._metadatos(
        train,
        test,
        validacion,
        random_state=random_state,
        particion=particion,
        opciones=opciones,
    )
    metadatos.update(
        {
            "estrategia_seleccion": "holdout",
            "metrica_seleccion": metrica,
            "validacion_externa": validacion_externa,
            "validation_size_sobre_train": None
            if validacion_externa
            else validation_size,
            "reajustado_train_validacion": False,
            "test_usado_para_seleccion": False,
            "desempate": "primera configuracion en el orden declarado; luego primer algoritmo",
            "algoritmos": list(catalogo),
            "configuraciones": catalogo,
            "muestra_train": len(train),
            "muestra_validacion": len(validacion),
            "muestra_test": len(test),
            "distribuciones": {
                nombre: base.loc[indices, clasificador.target].value_counts().to_dict()
                for nombre, indices in (
                    ("train", train),
                    ("validacion", validacion),
                    ("test", test),
                )
            },
        }
    )
    if "XGBoost" in catalogo:
        try:
            metadatos["versiones"]["xgboost"] = version("xgboost")
        except PackageNotFoundError:
            metadatos["versiones"]["xgboost"] = "no instalado"
    filas = []
    ganadores = {}
    for algoritmo, configs in catalogo.items():
        for config in configs:
            fila: dict[str, Any] = {c: None for c in COLUMNAS_TABLA}
            fila.update(
                algoritmo=algoritmo,
                configuracion=config["nombre"],
                parametros=dict(config["parametros"]),
                seleccionado=False,
                estado="ok",
                error=None,
            )
            try:
                modelo, preparados = clasificador._ajustar(
                    base,
                    train,
                    algoritmo,
                    random_state,
                    opciones,
                    config["parametros"],
                )
                validado = clasificador._resultado(
                    base,
                    train,
                    validacion,
                    algoritmo,
                    modelo,
                    preparados,
                    {"conjunto_evaluacion": "validacion"},
                )
                fila["parametros"] = validado.parametros
                scores = {
                    "accuracy": validado.metricas["accuracy"],
                    **{
                        f"{nombre}_macro": validado.metricas[nombre]["macro"]
                        for nombre in ("precision", "recall", "f1")
                    },
                }
                fila.update(
                    {f"{nombre}_validacion": valor for nombre, valor in scores.items()}
                )
                fila["puntuacion_validacion"] = scores[metrica]
                anterior = ganadores.get(algoritmo)
                if anterior is None or scores[metrica] > anterior["score"]:
                    ganadores[algoritmo] = {
                        "score": scores[metrica],
                        "config": config,
                        "fila": len(filas),
                        "modelo": modelo,
                        "preparados": preparados,
                        "metricas_validacion": validado.metricas,
                    }
            except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                fila.update(estado="error", error=f"{type(exc).__name__}: {exc}")
            filas.append(fila)

    # Finalize every selection BEFORE accessing test predictions/metrics.
    mejor_algoritmo = max(ganadores, key=lambda a: ganadores[a]["score"], default=None)
    metadatos["selecciones"] = {a: g["config"]["nombre"] for a, g in ganadores.items()}
    metadatos["mejor_algoritmo_por_validacion"] = mejor_algoritmo
    metadatos["errores_prueba"] = {}
    mejores = {}
    for algoritmo, ganador in ganadores.items():
        filas[ganador["fila"]]["seleccionado"] = True
        try:
            resultado = clasificador._resultado(
                base,
                train,
                test,
                algoritmo,
                ganador["modelo"],
                ganador["preparados"],
                {
                    **metadatos,
                    "conjunto_evaluacion": "test",
                    "configuracion": ganador["config"]["nombre"],
                    "puntuacion_validacion": ganador["score"],
                    "metricas_validacion": ganador["metricas_validacion"],
                },
            )
            mejores[algoritmo] = resultado
        except (ValueError, TypeError, RuntimeError) as exc:
            metadatos["errores_prueba"][algoritmo] = f"{type(exc).__name__}: {exc}"
    if mejor_algoritmo in mejores:
        clasificador.resultado = mejores[mejor_algoritmo]
        clasificador.modelo = ganadores[mejor_algoritmo]["modelo"]
        clasificador._preparados = ganadores[mejor_algoritmo]["preparados"]
        clasificador.exactitud = clasificador.resultado.metricas["accuracy"]
    tabla = pd.DataFrame(filas, columns=COLUMNAS_TABLA)
    metadatos["candidatos_exitosos"] = int((tabla["estado"] == "ok").sum())
    metadatos["candidatos_fallidos"] = int((tabla["estado"] == "error").sum())
    return ResultadoExperimentoClasificacion(
        tabla=tabla,
        mejores=mejores,
        mejor_algoritmo=mejor_algoritmo,
        metadatos=metadatos,
    )
