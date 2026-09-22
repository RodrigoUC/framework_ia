"""
DataFrame — encapsulación simple y funcional de pandas.DataFrame.

La clase concentra operaciones tabulares y de EDA que pueden reutilizarse
desde consola, notebooks o Streamlit sin depender de ninguna interfaz.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DataFrame:
    """Encapsula un :class:`pandas.DataFrame` con operaciones simples de EDA."""

    def __init__(self, dataframe: pd.DataFrame | None = None) -> None:
        if dataframe is None:
            dataframe = pd.DataFrame()
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe debe ser una instancia de pandas.DataFrame.")
        self.datos = dataframe.copy()

    # ==========================================================
    # Representación
    # ==========================================================

    def __str__(self) -> str:
        return (
            "Clase DataFrame\n"
            f"Filas: {len(self.datos)}\n"
            f"Columnas: {self.datos.shape[1]}\n"
            f"Nombres: {list(self.datos.columns)}"
        )

    # ==========================================================
    # Carga y guardado
    # ==========================================================

    def cargar_csv(self, ruta, separador: str = ",", encoding: str = "utf-8"):
        """Carga un archivo CSV y lo asigna al atributo ``datos``."""
        ruta = Path(ruta)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo '{ruta}'.")
        self.datos = pd.read_csv(ruta, sep=separador, encoding=encoding)
        return self.datos

    def guardar_csv(self, ruta: str = "dataframe.csv"):
        """Guarda el DataFrame actual en un archivo CSV."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.datos.to_csv(ruta, index=False, encoding="utf-8")
        return ruta

    # ==========================================================
    # Inspección básica
    # ==========================================================

    def mostrar(self):
        """Retorna el DataFrame completo."""
        return self.datos

    def primeras_filas(self, n: int = 5):
        """Retorna las primeras ``n`` filas."""
        return self.datos.head(n)

    def ultimas_filas(self, n: int = 5):
        """Retorna las últimas ``n`` filas."""
        return self.datos.tail(n)

    def dimensiones(self):
        """Retorna la cantidad de filas y columnas."""
        return self.datos.shape

    def nombres_columnas(self):
        """Retorna los nombres de las columnas."""
        return list(self.datos.columns)

    def tipos_datos(self):
        """Retorna los tipos de datos de cada columna."""
        return self.datos.dtypes

    def contar_nulos(self):
        """Cuenta los valores faltantes por columna."""
        return self.datos.isnull().sum()

    # ==========================================================
    # Limpieza
    # ==========================================================

    def eliminar_duplicados(self):
        """Elimina registros duplicados y reconstruye el índice."""
        self.datos = self.datos.drop_duplicates().reset_index(drop=True)
        return self.datos

    def imputar_nulos(
        self,
        estrategia_numerica: str = "mediana",
        estrategia_categorica: str = "moda",
    ):
        """Imputa nulos: mediana/media en numéricas y moda en categóricas."""
        for columna in self.datos.columns:
            serie = self.datos[columna]
            if not serie.isna().any():
                continue

            if pd.api.types.is_numeric_dtype(serie):
                valor = (
                    serie.median()
                    if estrategia_numerica == "mediana"
                    else serie.mean()
                )
            else:
                moda = serie.mode(dropna=True)
                valor = moda.iloc[0] if not moda.empty else "Desconocido"
                if estrategia_categorica != "moda":
                    valor = "Desconocido"

            if pd.isna(valor):
                raise ValueError(
                    f"No fue posible imputar la columna '{columna}'."
                )
            self.datos[columna] = serie.fillna(valor)

        return self.datos

    def normalizar(self, columnas=None):
        """Escala columnas numéricas al rango [0, 1]."""
        for columna in self._seleccionar_numericas(columnas):
            minimo, maximo = self.datos[columna].min(), self.datos[columna].max()
            if maximo == minimo:
                self.datos[columna] = 0.0
            else:
                self.datos[columna] = (
                    self.datos[columna] - minimo
                ) / (maximo - minimo)
        return self.datos

    def estandarizar(self, columnas=None):
        """Escala columnas numéricas con media 0 y desviación 1."""
        for columna in self._seleccionar_numericas(columnas):
            media = self.datos[columna].mean()
            desviacion = self.datos[columna].std()
            if desviacion == 0 or pd.isna(desviacion):
                self.datos[columna] = 0.0
            else:
                self.datos[columna] = (self.datos[columna] - media) / desviacion
        return self.datos

    def seleccionar_variables(self, columnas=None, max_porcentaje_nulos: float = 50.0):
        """Retorna las columnas indicadas o las de buena calidad (sin nulos altos)."""
        if columnas is not None:
            inexistentes = [
                columna for columna in columnas
                if columna not in self.datos.columns
            ]
            if inexistentes:
                raise KeyError(f"Columnas inexistentes: {inexistentes}")
            return self.datos[list(columnas)].copy()

        seleccionadas = []
        for columna in self.datos.columns:
            porcentaje = self.datos[columna].isna().mean() * 100
            if porcentaje <= max_porcentaje_nulos and self.datos[columna].nunique() > 1:
                seleccionadas.append(columna)
        return self.datos[seleccionadas].copy()

    def _seleccionar_numericas(self, columnas=None) -> list[str]:
        """Valida y devuelve las columnas numéricas elegidas."""
        disponibles = self.datos.select_dtypes(include="number").columns.tolist()
        if columnas is None:
            return disponibles
        if not disponibles:
            raise ValueError("No hay columnas numéricas en el DataFrame.")
        inexistentes = [
            columna for columna in columnas if columna not in self.datos.columns
        ]
        if inexistentes:
            raise KeyError(f"Columnas inexistentes: {inexistentes}")
        no_numericas = [columna for columna in columnas if columna not in disponibles]
        if no_numericas:
            raise TypeError(f"Las columnas deben ser numéricas: {no_numericas}")
        return list(columnas)


    # ==========================================================
    # Análisis EDA
    # ==========================================================

    def distribucion_variables(self):
        """Resumen estadístico de variables numéricas y categóricas."""
        numericas = self.datos.select_dtypes(include="number")
        categoricas = self.datos.select_dtypes(exclude="number")
        resultado = {"numericas": pd.DataFrame(), "categoricas": pd.DataFrame()}

        if not numericas.empty:
            resumen = numericas.describe().T
            resumen["mediana"] = numericas.median()
            resumen["nulos"] = numericas.isna().sum()
            resumen["porcentaje_nulos"] = (numericas.isna().mean() * 100).round(2)
            resultado["numericas"] = resumen

        if not categoricas.empty:
            resumen = categoricas.describe().T
            resumen["nulos"] = categoricas.isna().sum()
            resumen["porcentaje_nulos"] = (categoricas.isna().mean() * 100).round(2)
            resultado["categoricas"] = resumen

        return resultado

    def frecuencias(self, columnas=None):
        """Frecuencias absolutas y relativas por columna."""
        columnas = self.datos.columns.tolist() if columnas is None else list(columnas)
        resultado = {}
        for columna in columnas:
            conteos = self.datos[columna].value_counts(dropna=False)
            resultado[columna] = pd.DataFrame(
                {
                    "frecuencia": conteos,
                    "porcentaje": (conteos / len(self.datos) * 100).round(2),
                }
            )
        return resultado

    def histogramas(self, columnas=None, mostrar: bool = True):
        """Histogramas (con KDE) de cada variable numérica."""
        numericas = self._seleccionar_numericas(columnas)
        cantidad = len(numericas)
        if cantidad == 0:
            raise ValueError("No hay columnas numéricas para graficar.")

        filas = int(np.ceil(cantidad / 3))
        figura, ejes = plt.subplots(
            filas,
            min(cantidad, 3),
            figsize=(12, 3.5 * filas),
            squeeze=False,
        )
        ejes = np.array(ejes).ravel()
        for eje, columna in zip(ejes, numericas):
            sns.histplot(data=self.datos, x=columna, kde=True, ax=eje, color="#2878B5")
            eje.set_title(f"Distribución de {columna}")
        for eje in ejes[cantidad:]:
            eje.axis("off")

        figura.tight_layout()
        if mostrar:
            plt.show()
        return figura, ejes[:cantidad]

    def boxplots(self, columnas=None, mostrar: bool = True):
        """Diagramas de caja de cada variable numérica."""
        numericas = self._seleccionar_numericas(columnas)
        cantidad = len(numericas)
        if cantidad == 0:
            raise ValueError("No hay columnas numéricas para graficar.")

        filas = int(np.ceil(cantidad / 3))
        figura, ejes = plt.subplots(
            filas,
            min(cantidad, 3),
            figsize=(12, 3.0 * filas),
            squeeze=False,
        )
        ejes = np.array(ejes).ravel()
        for eje, columna in zip(ejes, numericas):
            sns.boxplot(data=self.datos, x=columna, ax=eje, color="#7DBE76")
            eje.set_title(f"Diagrama de caja de {columna}")
        for eje in ejes[cantidad:]:
            eje.axis("off")

        figura.tight_layout()
        if mostrar:
            plt.show()
        return figura, ejes[:cantidad]


    def scatterplots(self, columnas=None, mostrar: bool = True):
        """Matriz de dispersión (pairplot) entre variables numéricas."""
        numericas = self._seleccionar_numericas(columnas)
        if len(numericas) < 2:
            raise ValueError("Se requieren al menos dos columnas numéricas.")
        grafico = sns.pairplot(
            self.datos[numericas],
            diag_kind="hist",
            corner=True,
        )
        if mostrar:
            plt.show()
        return grafico

    def mapa_calor(self, columnas=None, metodo: str = "pearson", mostrar: bool = True):
        """Grafica y retorna la matriz de correlación de variables numéricas."""
        numericas = self._seleccionar_numericas(columnas)
        correlacion = self.datos[numericas].corr(method=metodo)
        figura, eje = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            correlacion,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=eje,
        )
        eje.set_title(f"Correlación ({metodo})")
        figura.tight_layout()
        if mostrar:
            plt.show()
        return correlacion.round(3)

    def detectar_outliers(self, columnas=None, factor_iqr: float = 1.5):
        """Detecta valores atípicos mediante límites de rango intercuartílico."""
        numericas = self._seleccionar_numericas(columnas)
        resumen = []
        filas_outliers = {}
        mascara_global = pd.Series(False, index=self.datos.index)

        for columna in numericas:
            q1, q3 = self.datos[columna].quantile([0.25, 0.75])
            iqr = q3 - q1
            limite_inferior = q1 - factor_iqr * iqr
            limite_superior = q3 + factor_iqr * iqr
            mascara = self.datos[columna].lt(limite_inferior) | self.datos[
                columna
            ].gt(limite_superior)
            mascara = mascara.fillna(False)
            mascara_global |= mascara
            cantidad = int(mascara.sum())

            resumen.append(
                {
                    "columna": columna,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "limite_inferior": limite_inferior,
                    "limite_superior": limite_superior,
                    "cantidad_outliers": cantidad,
                    "porcentaje_outliers": round(cantidad / len(self.datos) * 100, 2),
                }
            )
            filas_outliers[columna] = self.datos.loc[mascara].copy()

        return {
            "resumen": pd.DataFrame(resumen).set_index("columna"),
            "filas_outliers": filas_outliers,
            "filas_con_algun_outlier": self.datos.loc[mascara_global].copy(),
        }
