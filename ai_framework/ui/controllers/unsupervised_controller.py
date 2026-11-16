"""Compatibility imports for the unsupervised operations."""

from ai_framework.models.unsupervised.operations import (
    _numeric_columns as _numeric_columns,
    _categorical_columns as _categorical_columns,
    _sample_data as _sample_data,
    _create_reducer as _create_reducer,
    _create_cluster as _create_cluster,
    run_pca as run_pca,
    run_kmeans as run_kmeans,
    evaluate_kmeans_k as evaluate_kmeans_k,
    run_kmedoids as run_kmedoids,
    run_hac as run_hac,
    compare_hac_methods as compare_hac_methods,
    run_tsne as run_tsne,
    run_umap as run_umap,
)

# Backward compatibility aliases
ejecutar_acp = run_pca
ejecutar_kmeans = run_kmeans
evaluar_kmeans_k = evaluate_kmeans_k
ejecutar_kmedoids = run_kmedoids
ejecutar_hac = run_hac
comparar_metodos_hac = compare_hac_methods
ejecutar_tsne = run_tsne
ejecutar_umap = run_umap

# Backward compatibility aliases
_columnas_numericas = _numeric_columns
_columnas_categoricas = _categorical_columns
_muestrear = _sample_data
_crear_reductor = _create_reducer
_crear_cluster = _create_cluster
