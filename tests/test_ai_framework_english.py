"""Unit tests verifying the English public API of ai_framework."""

import unittest
import pandas as pd
import numpy as np

from ai_framework import (
    Classification,
    Regression,
    Clustering,
    DataFrame,
    EDA,
    CSVLoader,
    CSVConfig,
    DataSplitter,
    PartitionConfig,
    SupervisedPreprocessor,
    UnsupervisedPreprocessor,
    PreprocessingConfig,
    ClassificationResult,
    RegressionResult,
    ClusterResult,
    PartitionResult,
)
from ai_framework.datasets import diabetes, concrete, potability


class TestAIFrameworkEnglish(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 3))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        self.df_clf = pd.DataFrame(x, columns=["f1", "f2", "f3"])
        self.df_clf["target"] = y

        self.df_reg = pd.DataFrame(x, columns=["f1", "f2", "f3"])
        self.df_reg["target"] = x[:, 0] * 2.0 + x[:, 1] * 0.5 + rng.normal(scale=0.1, size=50)

    def test_dataframe_and_eda(self):
        df_wrapped = DataFrame(self.df_clf)
        self.assertEqual(len(df_wrapped.datos), 50)
        eda = EDA(self.df_clf)
        quality = eda.resumen_calidad()
        self.assertIn("filas", quality)
        self.assertEqual(quality["filas"], 50)

    def test_classification_english(self):
        clf = Classification(self.df_clf, target="target")
        res = clf.entrenar(algoritmo="KNN", n_neighbors=3)
        self.assertIsInstance(res, ClassificationResult)
        self.assertIn("accuracy", res.metricas)
        self.assertIn("f1", res.metricas)

    def test_regression_english(self):
        reg = Regression(self.df_reg, target="target")
        res = reg.entrenar(algoritmo="RLM")
        self.assertIsInstance(res, RegressionResult)
        self.assertIn("rmse", res.metricas)
        self.assertIn("r2", res.metricas)

    def test_clustering_english(self):
        clusterer = Clustering(self.df_clf[["f1", "f2", "f3"]])
        res = clusterer.K_means(n_clusters=3, random_state=42)
        self.assertIsInstance(res, ClusterResult)
        self.assertEqual(len(res.etiquetas), 50)

    def test_partitioning_english(self):
        splitter = DataSplitter(configuracion=PartitionConfig(porcentaje_test=0.2, semilla=42))
        res = splitter.dividir(self.df_clf)
        self.assertIsInstance(res, PartitionResult)
        self.assertEqual(len(res.train) + len(res.test), 50)

    def test_datasets_english(self):
        df_diab, _, meta_diab = diabetes.cargar_diabetes_regresion()
        self.assertGreater(len(df_diab), 0)
        self.assertIn("sha256", meta_diab)

        self.assertIn("cement", concrete.PREDICTORAS_CONCRETE)
        self.assertEqual(concrete.OBJETIVO_CONCRETE, "strength")
        self.assertIn("ph", potability.PREDICTORAS_POTABILIDAD)
        self.assertEqual(potability.OBJETIVO_POTABILIDAD, "Potability")


if __name__ == "__main__":
    unittest.main()
