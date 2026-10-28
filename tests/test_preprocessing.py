"""Pruebas exhaustivas del módulo funcional framework_ia.datos.preprocesamiento."""

import numpy as np
import pandas as pd
import pytest

from framework_ia.datos.preprocesamiento import (
    ConfiguracionPreprocesamiento,
    PreprocesadorNoSupervisado,
    PreprocesadorSupervisado,
)
from framework_ia.resultados import DatosPreparados


def test_preprocesador_no_supervisado_numerico_estandar():
    datos = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "constante": [7.0, 7.0, 7.0, 7.0, 7.0],
    })
    cfg = ConfiguracionPreprocesamiento(imputar=True, estandarizar=True)
    prep = PreprocesadorNoSupervisado(cfg)
    res = prep.ajustar_transformar(datos)

    assert isinstance(res, DatosPreparados)
    assert res.columnas_descartadas == ("constante",)
    assert "num1" in res.matriz.columns
    assert "num2" in res.matriz.columns
    assert not res.matriz.isna().any().any()
    assert np.isclose(res.matriz["num2"].mean(), 0.0, atol=1e-7)


def test_preprocesador_no_supervisado_categoricas():
    datos = pd.DataFrame({
        "edad": [25, 30, 35, 40],
        "ciudad": ["Bogotá", "Medellín", "Bogotá", "Cali"],
    })
    cfg_rechazo = ConfiguracionPreprocesamiento(
        columnas=["edad", "ciudad"], incluir_categoricas=False
    )
    with pytest.raises(TypeError, match="columnas categóricas"):
        PreprocesadorNoSupervisado(cfg_rechazo).ajustar_transformar(datos)

    cfg_auto_filtro = ConfiguracionPreprocesamiento(columnas=None, incluir_categoricas=False)
    res_auto = PreprocesadorNoSupervisado(cfg_auto_filtro).ajustar_transformar(datos)
    assert res_auto.matriz.columns.tolist() == ["edad"]

    cfg_valida = ConfiguracionPreprocesamiento(incluir_categoricas=True)
    res = PreprocesadorNoSupervisado(cfg_valida).ajustar_transformar(datos)
    assert res.matriz.shape[0] == 4
    assert res.matriz.shape[1] > 2
    assert not res.matriz.isna().any().any()


def test_preprocesador_no_supervisado_columnas_inexistentes_o_vacias():
    datos = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cfg_col = ConfiguracionPreprocesamiento(columnas=["inexistente"])
    with pytest.raises(KeyError, match="Columnas inexistentes"):
        PreprocesadorNoSupervisado(cfg_col).ajustar_transformar(datos)

    with pytest.raises(ValueError, match="al menos una fila"):
        PreprocesadorNoSupervisado(ConfiguracionPreprocesamiento()).ajustar_transformar(pd.DataFrame())


def test_preprocesador_supervisado_clasificacion_con_seleccion_k():
    X = pd.DataFrame({
        "util": [1.0, 2.0, 1.1, 2.1, 1.2, 2.2],
        "ruido": [10.0, 5.0, 12.0, 3.0, 8.0, 6.0],
        "const": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }, index=[10, 11, 12, 13, 14, 15])
    y = pd.Series(["A", "B", "A", "B", "A", "B"], index=X.index)

    prep = PreprocesadorSupervisado(seleccion_k=1, random_state=42)
    prep.fit(X, y)
    out = prep.transform(X)

    assert prep.columnas_descartadas_ == ("const",)
    assert out.index.tolist() == X.index.tolist()
    assert len(out.columns) == 1
    assert "util" in out.columns[0]

    ranking = prep.puntajes_informacion_mutua()
    assert ranking is not None
    assert len(ranking) == 2


def test_preprocesador_supervisado_regresion_con_seleccion_k():
    X = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [0.5, 0.2, 0.9, 0.1, 0.4, 0.8],
    })
    y_continuo = pd.Series([2.1, 4.0, 6.2, 8.1, 9.9, 12.1])

    prep = PreprocesadorSupervisado(seleccion_k=1, random_state=0)
    prep.fit(X, y_continuo)
    out = prep.transform(X)

    assert len(out.columns) == 1
    assert "f1" in out.columns[0]


def test_preprocesador_supervisado_imputar_false_falla_si_hay_nulos():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    prep = PreprocesadorSupervisado(imputar=False)
    with pytest.raises(ValueError, match="valores faltantes"):
        prep.fit(X)


def test_preprocesador_supervisado_ajustar_transformar():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    prep = PreprocesadorSupervisado()
    res = prep.ajustar_transformar(X)

    assert isinstance(res, DatosPreparados)
    assert res.matriz.shape == (3, 2)
    assert res.columnas_origen == ("a", "b")
