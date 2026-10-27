"""Validación de artefactos sintéticos; estos casos no son resultados académicos."""

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from framework_ia import Regresion
from framework_ia.evidencia_lab03 import EvidenciaLab03, cargar_evidencia
from framework_ia.exportacion_lab03 import construir_archivos_experimento
from framework_ia.utils.configuracion_regresion import ALGORITMOS_LAB3


@pytest.fixture(scope="module")
def artefactos(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("evidencia-base")
    rng = np.random.default_rng(71)
    datos = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    datos["respuesta"] = 5 * datos.a - 2 * datos.b + rng.normal(size=len(datos))
    csv = raiz / "observaciones.csv"
    csv.write_text(datos.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    datos = pd.read_csv(csv)
    original = raiz / "fuente.xls"
    original.write_bytes(b"Fixture de procedencia; no es un libro de calculo real.")
    proc = {
        "archivo": csv.name,
        "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "filas": len(datos),
        "columnas": datos.columns.tolist(),
        "fuente": "Fixture sintetico de software",
        "lectura": {"separador": ",", "decimal": ".", "encoding": "utf-8"},
        "original": {
            "archivo": original.name,
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        },
    }
    (raiz / "procedencia.json").write_text(json.dumps(proc), encoding="utf-8")
    resultado = Regresion(dataframe=datos, target="respuesta").experimentar(
        feature_simple="b"
    )
    argumentos = {"target": "respuesta", "semilla": 42, "folds": 5, "test_size": 0.25}
    archivos = construir_archivos_experimento(
        resultado,
        argumentos=argumentos,
        dataset={k: v for k, v in proc.items() if k != "original"},
    )
    for nombre, contenido in archivos.items():
        destino = raiz / "resultados" / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
    return raiz


@pytest.fixture
def evidencia(artefactos, tmp_path):
    destino = tmp_path / "datos"
    shutil.copytree(artefactos, destino)
    return destino


def cargar(raiz):
    return cargar_evidencia(
        raiz / "resultados", raiz / "observaciones.csv", raiz / "procedencia.json"
    )


def editar_json(ruta, cambiar):
    objeto = json.loads(ruta.read_text(encoding="utf-8"))
    cambiar(objeto)
    ruta.write_text(json.dumps(objeto), encoding="utf-8")


def editar_tabla(raiz, cambiar):
    ruta = raiz / "resultados" / "comparacion_validacion.csv"
    tabla = pd.read_csv(ruta)
    cambiar(tabla)
    tabla.to_csv(ruta, index=False)


def test_carga_nueve_modelos_catorce_candidatos_sin_ajustes_ni_escrituras(evidencia):
    antes = {p: p.read_bytes() for p in evidencia.rglob("*") if p.is_file()}
    with (
        patch.object(Regresion, "_ajustar", side_effect=AssertionError("No entrenar")),
        patch.object(Path, "write_bytes", side_effect=AssertionError("No escribir")),
        patch.object(Path, "write_text", side_effect=AssertionError("No escribir")),
    ):
        resultado = cargar(evidencia)
    assert isinstance(resultado, EvidenciaLab03)
    assert len(resultado.comparacion) == 14
    assert tuple(resultado.mejores) == ALGORITMOS_LAB3
    assert len(resultado.hashes) == 42
    for algoritmo, conjuntos in resultado.predicciones.items():
        assert list(conjuntos) == ["train", "test"]
        assert (
            conjuntos["test"].index.tolist()
            == resultado.configuracion["experimento"]["test_indices"]
        )
        assert_frame_equal(
            conjuntos["test"],
            pd.read_csv(
                evidencia / "resultados" / algoritmo / "predicciones_test.csv",
                index_col="fila",
            ),
        )
    assert antes == {p: p.read_bytes() for p in evidencia.rglob("*") if p.is_file()}


def test_procedencia_generica_sin_original_es_valida(evidencia):
    editar_json(evidencia / "procedencia.json", lambda p: p.pop("original"))
    assert len(cargar(evidencia).hashes) == 41


def test_recalcula_metricas_aunque_snapshot_y_resumen_coincidan(evidencia):
    carpeta = evidencia / "resultados"
    editar_json(
        carpeta / "RLM" / "metricas.json",
        lambda p: p["test"].__setitem__("rmse", 123.0),
    )
    editar_json(
        carpeta / "mejores_modelos.json",
        lambda p: p["RLM"]["metricas_test"].__setitem__("rmse", 123.0),
    )
    with pytest.raises(ValueError, match="RLM/test/rmse"):
        cargar(evidencia)


def test_rechaza_infinito_por_exponente_json(evidencia):
    ruta = evidencia / "procedencia.json"
    contenido = ruta.read_text(encoding="utf-8")
    ruta.write_text(contenido[:-1] + ', "extra": 1e999}', encoding="utf-8")
    with pytest.raises(ValueError, match="no finitos"):
        cargar(evidencia)


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("semilla", 1),
        ("folds", 3),
        ("test_usado_para_seleccion", True),
        ("candidatos_fallidos", 1),
        ("estrategia_seleccion", "holdout"),
        ("mejor_algoritmo_por_validacion", "no existe"),
    ],
)
def test_rechaza_protocolo_o_seleccion_alterados(evidencia, campo, valor):
    editar_json(
        evidencia / "resultados" / "configuracion.json",
        lambda p: p["experimento"].__setitem__(campo, valor),
    )
    with pytest.raises(ValueError):
        cargar(evidencia)


@pytest.mark.parametrize(
    "alteracion", ["test", "pliegue", "trainonly", "catalogo", "versiones"]
)
def test_rechaza_particiones_preparacion_y_catalogo_alterados(evidencia, alteracion):
    def cambiar(config):
        meta = config["experimento"]
        if alteracion == "test":
            meta["test_indices"][0] = meta["train_indices"][0]
        elif alteracion == "pliegue":
            meta["pliegues"][0]["train_indices"].append(meta["test_indices"][0])
        elif alteracion == "trainonly":
            meta["preprocesamiento"]["ajustado_solo_train"] = False
        elif alteracion == "catalogo":
            meta["configuraciones"]["SVR"][0]["parametros"]["C"] = 999
        else:
            config["versiones"]["python"] = "inventado"

    editar_json(evidencia / "resultados" / "configuracion.json", cambiar)
    with pytest.raises(ValueError):
        cargar(evidencia)


@pytest.mark.parametrize(
    "alteracion", ["std", "media", "parametro", "ganador", "pliegues", "infinito"]
)
def test_rechaza_comparacion_inconsistente(evidencia, alteracion):
    def cambiar(tabla):
        if alteracion == "std":
            tabla.loc[0, "puntuacion_validacion_std"] += 1
        elif alteracion == "media":
            tabla.loc[0, "mae_validacion"] += 1
        elif alteracion == "parametro":
            i = tabla.index[tabla.algoritmo == "SVR"][0]
            parametros = json.loads(tabla.loc[i, "parametros"])
            parametros["C"] = 999
            tabla.loc[i, "parametros"] = json.dumps(parametros)
        elif alteracion == "ganador":
            tabla.loc[0, "seleccionado"] = False
        elif alteracion == "pliegues":
            tabla.loc[0, "metricas_pliegues"] = "[]"
        else:
            tabla.loc[0, "rmse_validacion"] = np.inf

    editar_tabla(evidencia, cambiar)
    with pytest.raises(ValueError):
        cargar(evidencia)


@pytest.mark.parametrize(
    "alteracion", ["real", "residuo", "indice", "duplicado", "no_finito"]
)
def test_rechaza_predicciones_incompatibles(evidencia, alteracion):
    ruta = evidencia / "resultados" / "RLM" / "predicciones_test.csv"
    tabla = pd.read_csv(ruta)
    if alteracion == "indice":
        tabla.loc[0, "fila"] = 100000
    elif alteracion == "duplicado":
        tabla.loc[0, "fila"] = tabla.loc[1, "fila"]
    elif alteracion == "no_finito":
        tabla.loc[0, "prediccion"] = np.nan
    else:
        tabla.loc[0, alteracion] += 1
    tabla.to_csv(ruta, index=False)
    with pytest.raises(ValueError):
        cargar(evidencia)


@pytest.mark.parametrize("archivo", ["metricas", "snapshot", "interna", "resumen"])
def test_rechaza_snapshots_y_metricas_alterados(evidencia, archivo):
    carpeta = evidencia / "resultados"
    if archivo == "metricas":
        editar_json(
            carpeta / "RLM" / "metricas.json",
            lambda p: p["test"].__setitem__("rmse", 500),
        )
    elif archivo == "snapshot":
        editar_json(
            carpeta / "RLM" / "configuracion.json",
            lambda p: p["modelo"]["metadatos"]["train_indices"].reverse(),
        )
    elif archivo == "interna":
        editar_json(
            carpeta / "LassoCV" / "configuracion.json",
            lambda p: p["modelo"]["metadatos"]["cv_interna"].__setitem__(
                "preprocesamiento_por_pliegue", True
            ),
        )
    else:
        editar_json(carpeta / "mejores_modelos.json", lambda p: p.pop("RF"))
    with pytest.raises(ValueError):
        cargar(evidencia)


@pytest.mark.parametrize(
    "alteracion", ["hash", "filas", "columnas", "lectura", "original", "ruta"]
)
def test_rechaza_procedencia_alterada(evidencia, alteracion):
    def cambiar(proc):
        if alteracion == "hash":
            proc["sha256"] = "0" * 64
        elif alteracion == "filas":
            proc["filas"] -= 1
        elif alteracion == "columnas":
            proc["columnas"].reverse()
        elif alteracion == "lectura":
            proc["lectura"]["decimal"] = ","
        elif alteracion == "original":
            proc["original"]["sha256"] = "0" * 64
        else:
            proc["original"]["archivo"] = "../externo.xls"

    editar_json(evidencia / "procedencia.json", cambiar)
    with pytest.raises(ValueError):
        cargar(evidencia)


@pytest.mark.parametrize(
    "alteracion", ["faltante", "json", "duplicado_json", "objeto", "csv"]
)
def test_normaliza_errores_de_entradas_malformadas(evidencia, alteracion):
    ruta = evidencia / "resultados" / "configuracion.json"
    if alteracion == "faltante":
        ruta.unlink()
    elif alteracion == "json":
        ruta.write_text("{", encoding="utf-8")
    elif alteracion == "duplicado_json":
        ruta.write_text('{"dataset": {}, "dataset": {}}', encoding="utf-8")
    elif alteracion == "objeto":
        ruta.write_text("[]", encoding="utf-8")
    else:
        (evidencia / "observaciones.csv").write_bytes(b"")
    with pytest.raises(ValueError):
        cargar(evidencia)
