"""Pruebas funcionales del flujo desacoplado de análisis no supervisado."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from framework_ia.datos.eda import EDA
from framework_ia.datos.fuentes import CargadorCSV, ConfiguracionCSV
from framework_ia.datos.particion import ConfiguracionParticion, Particionador
from framework_ia.datos.preprocesamiento import (
    ConfiguracionPreprocesamiento,
    PreprocesadorNoSupervisado,
)
from framework_ia.modelos.no_supervisado import (
    Cluster,
    NoSupervisado,
)

from framework_ia.modelos.supervisado import (
    Clasificacion,
    Regresion,
    Supervisado,
)
from framework_ia.visualizacion import VisualizadorNoSupervisado


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
        resultado = Cluster(
            dataframe=self.datos,
            features=["x", "y", "z"],
        ).ACP(n_componentes=2)
        self.assertEqual(resultado.coordenadas.shape, (30, 2))
        self.assertEqual(resultado.cargas.shape, (3, 2))
        figura = VisualizadorNoSupervisado.circulo_correlacion(resultado)
        self.assertIsInstance(figura, go.Figure)
        biplot = VisualizadorNoSupervisado.sobreposicion_acp(resultado)
        self.assertIsInstance(biplot, go.Figure)

    def test_acp_overlay_preserves_scores_and_spreads_left_right_labels(self) -> None:
        resultado = Cluster(
            dataframe=self.datos,
            features=["x", "y", "z"],
        ).ACP(n_componentes=2)
        coordenadas = resultado.coordenadas.copy()
        cargas_originales = resultado.cargas.copy()

        for cantidad in (4, 12, 20):
            variables = [f"variable_{indice:02}" for indice in range(cantidad)]
            lados = [1 if indice % 2 == 0 else -1 for indice in range(cantidad)]
            cargas = pd.DataFrame(
                {
                    resultado.cargas.columns[0]: [
                        lado * (0.88 + indice / (cantidad * 20))
                        for indice, lado in enumerate(lados)
                    ],
                    resultado.cargas.columns[1]: np.linspace(-0.015, 0.015, cantidad),
                },
                index=variables,
            )
            figura = VisualizadorNoSupervisado.sobreposicion_acp(
                replace(resultado, cargas=cargas)
            )

            escala = coordenadas.abs().max(axis=0).replace(0, 1)
            puntos = coordenadas.iloc[:, :2].divide(escala.iloc[:2], axis="columns")
            np.testing.assert_allclose(figura.data[0].x, puntos.iloc[:, 0])
            np.testing.assert_allclose(figura.data[0].y, puntos.iloc[:, 1])

            flechas = [
                anotacion
                for anotacion in figura.layout.annotations
                if anotacion.showarrow
            ]
            self.assertEqual(
                [(anotacion.x, anotacion.y) for anotacion in flechas],
                list(zip(cargas.iloc[:, 0], cargas.iloc[:, 1], strict=True)),
            )
            etiquetas = {
                anotacion.text: anotacion
                for anotacion in figura.layout.annotations
                if not anotacion.showarrow
            }
            self.assertEqual(set(etiquetas), set(variables))
            self.assertTrue(any(shape.type == "circle" for shape in figura.layout.shapes))
            self.assertEqual(figura.layout.yaxis.scaleanchor, "x")
            self.assertEqual(figura.layout.yaxis.scaleratio, 1)

            for lado in (-1, 1):
                grupo = [
                    etiquetas[variable]
                    for indice, variable in enumerate(variables)
                    if lados[indice] == lado
                ]
                self.assertTrue(
                    all(
                        anotacion.xanchor == ("left" if lado > 0 else "right")
                        for anotacion in grupo
                    )
                )

            circulo = VisualizadorNoSupervisado.circulo_correlacion(resultado)
            geometria = next(
                shape for shape in circulo.layout.shapes if shape.type == "circle"
            )
            self.assertEqual(
                (geometria.x0, geometria.y0, geometria.x1, geometria.y1), (-1, -1, 1, 1)
            )
            pd.testing.assert_frame_equal(coordenadas, resultado.coordenadas)
            pd.testing.assert_frame_equal(cargas_originales, resultado.cargas)

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
        resultado = Cluster(
            dataframe=self.datos,
            features=["x", "y", "z"],
        ).TSNE(perplexity=5, max_iter=250)
        self.assertEqual(resultado.coordenadas.shape, (30, 2))

    def test_umap_produce_dos_dimensiones(self) -> None:
        """UMAP sustituye una ruta de caché inutilizable en cualquier sistema."""
        with TemporaryDirectory(prefix="framework_numba_test_") as temporal:
            # Un archivo no puede usarse como carpeta; /proc sólo funciona en Unix.
            ruta_invalida = Path(temporal) / "cache_como_archivo"
            ruta_invalida.write_text("fixture", encoding="utf-8")
            with patch.dict("os.environ", {"NUMBA_CACHE_DIR": str(ruta_invalida)}):
                resultado = Cluster(
                    dataframe=self.datos,
                    features=["x", "y", "z"],
                ).UMAP(n_neighbors=5, min_dist=0.1)
                self.assertNotEqual(os.environ["NUMBA_CACHE_DIR"], str(ruta_invalida))
        self.assertEqual(resultado.coordenadas.shape, (30, 2))
        self.assertFalse(resultado.coordenadas.isna().any().any())

    def test_arquitectura_declara_clases_y_metodos(self) -> None:
        """Las extensiones existen como contratos utilizables de punta a punta."""
        self.assertTrue(issubclass(NoSupervisado, EDA))
        self.assertTrue(issubclass(Supervisado, EDA))
        self.assertTrue(issubclass(Clasificacion, Supervisado))
        self.assertTrue(issubclass(Regresion, Supervisado))
        self.assertTrue(issubclass(Cluster, NoSupervisado))

        for clase, metodos in (
            (Clasificacion, ("RF", "NR", "RL")),
            (Regresion, ("RLS", "RLM")),
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
        previsualizacion = clasif.previsualizar_particion(
            test_size=0.25,
            random_state=1,
            stratify=True,
        )
        self.assertEqual(previsualizacion["tam_train"], 30)
        self.assertEqual(previsualizacion["tam_test"], 10)
        datos_regresion = pd.DataFrame({"x": range(20), "target": np.arange(20) * 2})
        regresion = Regresion(dataframe=datos_regresion).RLS()
        self.assertEqual(regresion.algoritmo, "RLS")
        self.assertAlmostEqual(regresion.metricas["r2"], 1.0)
        resultado_rl = clasif.regresion_logistica(test_size=0.25, random_state=1)
        self.assertEqual(resultado_rl.algoritmo, "RL")
        self.assertIn("accuracy", resultado_rl.metricas)
        base_sup = Supervisado(dataframe=datos_regresion)
        res_base = base_sup.entrenar(algoritmo="RLS")
        self.assertEqual(res_base.algoritmo, "RLS")
        self.assertIs(base_sup.evaluar(), res_base)
        self.assertEqual(len(base_sup.predecir()), len(res_base.y_pred))

    def test_agrupamiento_y_reduccion_no_supervisada(self) -> None:
        cluster = Cluster(dataframe=self.datos, features=["x", "y", "z"])
        self.assertEqual(cluster.K_means(n_clusters=2).etiquetas.nunique(), 2)
        reduccion = Cluster(
            dataframe=self.datos,
            features=["x", "y", "z"],
        )
        self.assertEqual(reduccion.ACP(n_componentes=2).coordenadas.shape, (30, 2))

    def test_resumen_columnas_identifica_tipos_numerica_y_categorica(self) -> None:
        tabla = EDA(dataframe=self.datos).resumen_columnas()
        self.assertEqual(tabla.loc["x", "tipo"], "Numérica")
        self.assertEqual(tabla.loc["categoria", "tipo"], "Categórica")
        self.assertEqual(int(tabla.loc["x", "nulos"]), 1)
        self.assertEqual(int(tabla.loc["categoria", "valores_unicos"]), 2)

    def test_eliminar_columnas_valida_existencia_y_totalidad(self) -> None:
        eda = EDA(dataframe=self.datos)
        eda.eliminar_columnas(["categoria"])
        self.assertNotIn("categoria", eda.datos.columns)
        with self.assertRaises(KeyError):
            eda.eliminar_columnas(["no_existe"])
        with self.assertRaises(ValueError):
            eda.eliminar_columnas(list(eda.datos.columns))

    def test_particionador_divide_train_test_validacion_estratificado(self) -> None:
        configuracion = ConfiguracionParticion(
            porcentaje_test=0.2,
            porcentaje_validacion=0.2,
            columna_estratificacion="categoria",
            semilla=1,
        )
        resultado = Particionador(configuracion).dividir(self.datos)
        self.assertEqual(len(resultado.train), 18)
        self.assertEqual(len(resultado.test), 6)
        self.assertIsNotNone(resultado.validacion)
        self.assertEqual(len(resultado.validacion), 6)
        self.assertIsNone(resultado.advertencia)
        self.assertEqual(sum(resultado.distribucion_train.values()), 18)
        self.assertEqual(sum(resultado.distribucion_test.values()), 6)
        self.assertTrue(
            set(resultado.train.index)
            .isdisjoint(resultado.test.index)
        )
        self.assertTrue(set(resultado.train.index).isdisjoint(resultado.validacion.index))

    def test_clasificacion_reutiliza_particion_externa(self) -> None:
        datos_clasificacion = pd.DataFrame(
            {
                "feat1": np.arange(40),
                "feat2": np.arange(40) * 2 + 1,
                "target": ["clase_a", "clase_b"] * 20,
            }
        )
        particion = Particionador(
            ConfiguracionParticion(
                porcentaje_test=0.25,
                columna_estratificacion="target",
                semilla=7,
            )
        ).dividir(datos_clasificacion)

        clasif = Clasificacion(dataframe=datos_clasificacion, target="target")
        resultado = clasif.RF(n_estimators=20, particion=particion)
        self.assertEqual(resultado.muestra_train, len(particion.train))
        self.assertEqual(resultado.muestra_test, len(particion.test))


if __name__ == "__main__":
    unittest.main()
