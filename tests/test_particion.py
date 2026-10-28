"""Pruebas exhaustivas del módulo funcional framework_ia.datos.particion."""

import numpy as np
import pandas as pd
import pytest

from framework_ia.datos.particion import (
    ConfiguracionParticion,
    Particionador,
    comprobar_clases,
    dividir_indices,
    huella_datos,
    validar_particion,
)
from framework_ia.resultados import ResultadoParticion


def test_huella_datos_consistencia_y_sensibilidad():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert huella_datos(df1) == huella_datos(df2)

    df_mod_val = pd.DataFrame({"a": [1, 2, 99], "b": [4.0, 5.0, 6.0]})
    assert huella_datos(df1) != huella_datos(df_mod_val)

    df_mod_idx = df1.copy()
    df_mod_idx.index = [10, 20, 30]
    assert huella_datos(df1) != huella_datos(df_mod_idx)

    df_mod_col = df1.rename(columns={"b": "c"})
    assert huella_datos(df1) != huella_datos(df_mod_col)

    df_mod_type = df1.copy()
    df_mod_type["a"] = df_mod_type["a"].astype(float)
    assert huella_datos(df1) != huella_datos(df_mod_type)


def test_particionador_division_train_test_correcta():
    datos = pd.DataFrame({
        "feat1": np.arange(100),
        "feat2": np.arange(100, 200),
    })
    cfg = ConfiguracionParticion(porcentaje_test=0.2, porcentaje_validacion=0.0, semilla=42)
    part = Particionador(cfg).dividir(datos)

    assert isinstance(part, ResultadoParticion)
    assert len(part.train) == 80
    assert len(part.test) == 20
    assert part.validacion is None
    assert set(part.train.index).isdisjoint(set(part.test.index))
    assert set(part.train.index).union(set(part.test.index)) == set(datos.index)
    validar_particion(part, datos)


def test_particionador_division_train_test_validacion():
    datos = pd.DataFrame({
        "x": np.arange(100),
        "target": [0, 1] * 50,
    })
    cfg = ConfiguracionParticion(
        porcentaje_test=0.2,
        porcentaje_validacion=0.1,
        columna_estratificacion="target",
        semilla=123,
    )
    part = Particionador(cfg).dividir(datos)

    assert len(part.train) == 70
    assert len(part.test) == 20
    assert len(part.validacion) == 10
    assert part.distribucion_train == {0: 35, 1: 35}
    assert part.distribucion_test == {0: 10, 1: 10}
    assert part.distribucion_validacion == {0: 5, 1: 5}

    validar_particion(part, datos)


def test_particionador_advertencia_por_nulos_o_clase_unica():
    datos_nulos = pd.DataFrame({"x": [1, 2, 3, 4], "cat": ["a", "a", np.nan, "b"]})
    cfg_nulos = ConfiguracionParticion(
        porcentaje_test=0.5,
        columna_estratificacion="cat",
    )
    part_nulos = Particionador(cfg_nulos).dividir(datos_nulos)
    assert part_nulos.advertencia is not None
    assert "nulos" in part_nulos.advertencia

    datos_unica = pd.DataFrame({"x": [1, 2, 3, 4], "cat": ["a", "a", "a", "a"]})
    cfg_unica = ConfiguracionParticion(
        porcentaje_test=0.5,
        columna_estratificacion="cat",
    )
    part_unica = Particionador(cfg_unica).dividir(datos_unica)
    assert part_unica.advertencia is not None
    assert "una sola clase" in part_unica.advertencia


def test_particionador_advertencia_clase_con_una_observacion():
    datos = pd.DataFrame({"x": [1, 2, 3, 4, 5], "cat": ["a", "a", "b", "b", "c"]})
    cfg = ConfiguracionParticion(porcentaje_test=0.4, columna_estratificacion="cat")
    part = Particionador(cfg).dividir(datos)
    assert part.advertencia is not None
    assert "sola observación" in part.advertencia


def test_particionador_validaciones_de_rangos_y_entradas():
    with pytest.raises(ValueError, match="al menos una fila"):
        Particionador(ConfiguracionParticion(porcentaje_test=0.2)).dividir(pd.DataFrame())

    with pytest.raises(ValueError, match="porcentaje de prueba"):
        Particionador(ConfiguracionParticion(porcentaje_test=0.0)).dividir(pd.DataFrame({"x": [1, 2]}))

    with pytest.raises(ValueError, match="porcentaje de prueba"):
        Particionador(ConfiguracionParticion(porcentaje_test=1.0)).dividir(pd.DataFrame({"x": [1, 2]}))

    with pytest.raises(ValueError, match="suma de prueba y validación"):
        Particionador(ConfiguracionParticion(porcentaje_test=0.6, porcentaje_validacion=0.5)).dividir(
            pd.DataFrame({"x": [1, 2]})
        )

    with pytest.raises(KeyError, match="no existe"):
        Particionador(
            ConfiguracionParticion(porcentaje_test=0.5, columna_estratificacion="inexistente")
        ).dividir(pd.DataFrame({"x": [1, 2]}))


def test_validar_particion_detecta_anomalias():
    datos = pd.DataFrame({"x": [10, 20, 30, 40]})
    part = Particionador(ConfiguracionParticion(porcentaje_test=0.5, semilla=0)).dividir(datos)

    with pytest.raises(TypeError, match="ResultadoParticion"):
        validar_particion("no es particion", datos)

    datos_dup = pd.DataFrame({"x": [10, 20, 30, 40]}, index=[0, 0, 1, 2])
    with pytest.raises(ValueError, match="índices únicos"):
        validar_particion(part, datos_dup)

    part_mod_val = ResultadoParticion(
        train=pd.DataFrame({"x": [999, 20]}, index=part.train.index),
        test=part.test,
        validacion=None,
        porcentaje_train=0.5,
        porcentaje_test=0.5,
        porcentaje_validacion=0.0,
        columna_estratificacion=None,
        distribucion_train=None,
        distribucion_test=None,
        distribucion_validacion=None,
        semilla=0,
    )
    with pytest.raises(ValueError, match="cambiaron los datos"):
        validar_particion(part_mod_val, datos)


def test_dividir_indices_y_comprobar_clases():
    df = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1]})
    train_idx, test_idx = dividir_indices(df, "y", test_size=0.5, random_state=42, stratify=True)
    assert len(train_idx) == 3
    assert len(test_idx) == 3

    with pytest.raises(ValueError, match="test_size"):
        dividir_indices(df, "y", test_size=1.5, random_state=42, stratify=False)

    comprobar_clases(pd.Series([0, 1, 0]), pd.Series([1, 0]))

    with pytest.raises(ValueError, match="al menos dos clases"):
        comprobar_clases(pd.Series([0, 0, 0]), pd.Series([0]))

    with pytest.raises(ValueError, match="ausentes de entrenamiento"):
        comprobar_clases(pd.Series([0, 1, 0]), pd.Series([2, 0]))
