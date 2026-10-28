from types import SimpleNamespace
import pandas as pd
from ai_framework.data.sources import ConfiguracionCSV
from ai_framework.ui.views.data_view import (
    _formatear_error_fuente,
    _verificar_calidad_carga_csv,
)


def test_formatear_error_fuente_parser_error():
    cfg = ConfiguracionCSV(separador=";")
    exc = pd.errors.ParserError("Error tokenizing data. C error: Expected 12 fields in line 3, saw 13")
    msg = _formatear_error_fuente(exc, cfg)
    assert "Error tokenizing data" not in msg
    assert "La estructura del archivo no coincide con el separador «;»" in msg
    assert "línea 3" in msg


def test_formatear_error_fuente_empty_data():
    cfg = ConfiguracionCSV(separador=",")
    exc = pd.errors.EmptyDataError("No columns to parse from file")
    msg = _formatear_error_fuente(exc, cfg)
    assert "vacío" in msg or "registros" in msg


def test_formatear_error_fuente_none_config():
    exc = ValueError("fixture failure")
    msg = _formatear_error_fuente(exc, None)
    assert "fixture failure" in msg


def test_verificar_calidad_carga_csv_detecta_coma():
    # Dataset with 1 column containing commas because separator was ';'
    df = pd.DataFrame({
        "id,nombre,edad,salario": [
            "1,Juan,30,2500",
            "2,Maria,25,3100",
            "3,Pedro,42,4200",
        ]
    })
    adv = _verificar_calidad_carga_csv(df, ";")
    assert adv is not None
    assert "1 sola columna" in adv
    assert "coma (,)" in adv


def test_verificar_calidad_carga_csv_multicolumn_ok():
    df = pd.DataFrame({
        "id": [1, 2],
        "nombre": ["A", "B"],
    })
    adv = _verificar_calidad_carga_csv(df, ",")
    assert adv is None
