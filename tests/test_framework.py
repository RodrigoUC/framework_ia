"""Pruebas funcionales del flujo desacoplado de análisis no supervisado."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from cluster import Cluster
from clasificacion import Clasificacion
from eda import EDA
from fuentes_datos import CargadorCSV, ConfiguracionCSV
from preprocesamiento import (
    ConfiguracionPreprocesamiento,
    PreprocesadorNoSupervisado,
)
from reduccion_dimensional import ReduccionDimensional
from no_supervisado import NoSupervisado
from progresion import Progresion
from supervisado import Supervisado
from visualizacion import VisualizadorNoSupervisado


class FrameworkNoSupervisadoTests(unittest.TestCase):
    """Comprueba que un dataset nuevo recorra el framework completo."""

    def setUp(self) -> None:
        generador = np.random.default_rng(42)
        grupo_a = generador.normal(loc=-2.0, scale=0.25, size=(15, 3))
        grupo_b = generador.normal(loc=2.0, scale=0.25, size=(15, 3))
        valores = np.vstack([grupo_a, grupo_b])
        self.datos = pd.DataFrame(valores, columns=["x", "y", "z"])
        self.datos["categoria"] = ["A"] * 15 + ["B"] * 15
        self.datos.loc[0, "x"] = np.nan

    def test_carga_csv_configurable(self) -> None:
        contenido = b"indice;valor\na;1,5\nb;2,5\n"
        configuracion = ConfiguracionCSV(
            separador=";",
            decimal=",",
            usar_primera_columna_como_indice=True,
        )
        resultado = CargadorCSV.cargar_bytes(contenido, configuracion)
        self.assertEqual(resultado.index.tolist(), ["a", "b"])
        self.assertAlmostEqual(resultado.loc["a", "valor"], 1.5)

    def test_preparacion_admite_nulos_y_categorias(self) -> None:
        configuracion = ConfiguracionPreprocesamiento(
            columnas=("x", "y", "categoria"),
            incluir_categoricas=True,
            estandarizar=True,
        )
        resultado = PreprocesadorNoSupervisado(configuracion).ajustar_transformar(
            self.datos
        )
        self.assertFalse(resultado.matriz.isna().any().any())
        self.assertGreaterEqual(resultado.matriz.shape[1], 4)

    def test_eda_prepara_dataset(self) -> None:
        duplicado = pd.concat([self.datos, self.datos.iloc[[1]]], ignore_index=True)
        resultado = EDA(dataframe=duplicado).preparar_dataset()
        self.assertEqual(len(resultado), 30)
        self.assertEqual(int(resultado.isna().sum().sum()), 0)

    def test_acp_retorna_componentes_y_figuras(self) -> None:
        resultado = ReduccionDimensional(
            dataframe=self.datos,
            features=["x", "y", "z"],
        ).ACP(n_componentes=2)
        self.assertEqual(resultado.coordenadas.shape, (30, 2))
        self.assertEqual(resultado.cargas.shape, (3, 2))
        figura = VisualizadorNoSupervisado.circulo_correlacion(resultado)
        self.assertIsNotNone(figura)
        plt.close(figura)
        biplot = VisualizadorNoSupervisado.sobreposicion_acp(resultado)
        self.assertIsNotNone(biplot)
        plt.close(biplot)

    def test_kmeans_kmedoids_y_hac_generan_clusters(self) -> None:
        modelo = Cluster(
            dataframe=self.datos,
            features=["x", "y", "z"],
        )
        kmeans = modelo.K_means(n_clusters=2)
        kmedoids = modelo.K_medoids(n_clusters=2)
        hac = modelo.HAC(n_clusters=2, metodo="ward")
        self.assertEqual(kmeans.etiquetas.nunique(), 2)
        self.assertEqual(kmedoids.etiquetas.nunique(), 2)
        self.assertEqual(hac.etiquetas.nunique(), 2)
        self.assertGreater(kmeans.silhouette, 0.5)
        self.assertGreater(kmedoids.silhouette, 0.5)
        self.assertIsNotNone(hac.matriz_vinculacion)

    def test_tsne_produce_dos_dimensiones(self) -> None:
        resultado = ReduccionDimensional(
            dataframe=self.datos,
            features=["x", "y", "z"],
        ).TSNE(perplexity=5, max_iter=250)
        self.assertEqual(resultado.coordenadas.shape, (30, 2))

    def test_arquitectura_declara_clases_y_metodos(self) -> None:
        """Las extensiones pendientes existen como contratos utilizables."""
        self.assertTrue(issubclass(NoSupervisado, EDA))
        self.assertTrue(issubclass(Supervisado, EDA))
        self.assertTrue(issubclass(Clasificacion, Supervisado))
        self.assertTrue(issubclass(Progresion, Supervisado))
        self.assertTrue(issubclass(Cluster, NoSupervisado))
        self.assertTrue(issubclass(ReduccionDimensional, NoSupervisado))

        for clase, metodos in (
            (Clasificacion, ("RF", "NR")),
            (Progresion, ("RLS", "RLM", "RL")),
        ):
            for metodo in metodos:
                self.assertTrue(callable(getattr(clase, metodo, None)))

        datos_clasificacion = pd.DataFrame(
            {
                "feat1": np.arange(40),
                "feat2": np.arange(40) * 2 + 1,
                "target": ["clase_a", "clase_b"] * 20,
            }
        )
        clasif = Clasificacion(dataframe=datos_clasificacion, target="target")
        resultado = clasif.RF(test_size=0.25, random_state=1, n_estimators=20)
        self.assertEqual(resultado.target, "target")
        self.assertEqual(resultado.algoritmo, "RF")
        self.assertEqual(len(resultado.y_true), len(resultado.y_pred))
        self.assertIn("accuracy", resultado.metricas)
        with self.assertRaises(NotImplementedError):
            Progresion(dataframe=self.datos, target="x").RLS()

    def test_aliases_pythonicos_conservan_la_interfaz_existente(self) -> None:
        cluster = Cluster(dataframe=self.datos, features=["x", "y", "z"])
        self.assertEqual(cluster.k_means(n_clusters=2).etiquetas.nunique(), 2)
        reduccion = ReduccionDimensional(
            dataframe=self.datos,
            features=["x", "y", "z"],
        )
        self.assertEqual(reduccion.acp(n_componentes=2).coordenadas.shape, (30, 2))


if __name__ == "__main__":
    unittest.main()
