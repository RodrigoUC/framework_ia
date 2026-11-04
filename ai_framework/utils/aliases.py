"""Compatibility aliases and translations for models and algorithms.

This module houses all legacy Spanish names and convenience aliases,
keeping the core model definitions in ai_framework/models clean and strictly English.
"""

from __future__ import annotations

from ai_framework.models.supervised.classification import Classification
from ai_framework.models.supervised.regression import Regression
from ai_framework.models.supervised.supervised import Supervised
from ai_framework.models.unsupervised.unsupervised import (
    OptionalDependencyError,
    Unsupervised,
)
from ai_framework.models.unsupervised.clustering import Clustering

# ==============================================================================
# Model class aliases (Spanish -> English)
# ==============================================================================
Clasificacion = Classification
Regresion = Regression
Supervisado = Supervised
NoSupervisado = Unsupervised
DependenciaOpcionalError = OptionalDependencyError
Cluster = Clustering

# ==============================================================================
# Classification method aliases and Spanish translations
# ==============================================================================
# Pythonic acronyms and case aliases
Classification.KNN = Classification.knn
Classification.DT = Classification.decision_tree
Classification.dt = Classification.decision_tree
Classification.RF = Classification.random_forest
Classification.rf = Classification.random_forest
Classification.XGBoost = Classification.xgboost
Classification.XGB = Classification.xgboost
Classification.xgb = Classification.xgboost
Classification.AdaBoost = Classification.adaboost
Classification.ADA = Classification.adaboost
Classification.ada = Classification.adaboost
Classification.NR = Classification.naive_bayes
Classification.nr = Classification.naive_bayes
Classification.RL = Classification.logistic_regression
Classification.rl = Classification.logistic_regression

# Spanish method names
Classification.arbol_decision = Classification.decision_tree
Classification.random_forest = Classification.random_forest
Classification.regresion_logistica = Classification.logistic_regression

# ==============================================================================
# Regression method aliases and Spanish translations
# ==============================================================================
# Pythonic acronyms and case aliases
Regression.RLS = Regression.simple_linear_regression
Regression.RLM = Regression.multiple_linear_regression
Regression.Lasso = Regression.lasso
Regression.LassoCV = Regression.lasso_cv
Regression.Ridge = Regression.ridge
Regression.RidgeCV = Regression.ridge_cv
Regression.SVR = Regression.svr
Regression.DT = Regression.decision_tree
Regression.dt = Regression.decision_tree
Regression.RF = Regression.random_forest
Regression.rf = Regression.random_forest

# Spanish method names
Regression.lineal_simple = Regression.simple_linear_regression
Regression.lineal_multiple = Regression.multiple_linear_regression
Regression.arbol_decision = Regression.decision_tree

# ==============================================================================
# Clustering method aliases
# ==============================================================================
Clustering.k_means = Clustering.K_means
Clustering.k_medoids = Clustering.K_medoids
Clustering.hac = Clustering.HAC
Clustering.acp = Clustering.ACP
Clustering.tsne = Clustering.TSNE
Clustering.umap = Clustering.UMAP

__all__ = [
    "Clasificacion",
    "Regresion",
    "Supervisado",
    "NoSupervisado",
    "DependenciaOpcionalError",
    "Cluster",
    "Classification",
    "Regression",
    "Supervised",
    "Unsupervised",
    "OptionalDependencyError",
    "Clustering",
]
