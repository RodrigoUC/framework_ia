"""User-facing texts of the agents in every supported language.

Findings, progress updates and fallback summaries are shown to people, so they
follow the interface language. Unknown languages fall back to Spanish, the
framework's default.
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "es"
LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}

MESSAGES: dict[str, dict[str, str]] = {
    "es": {
        # ---- agent names
        "agent_eda": "Agente EDA",
        "agent_clustering": "Agente de clustering",
        "agent_regression": "Agente de regresión",
        "agent_classification": "Agente de clasificación",
        # ---- base agent
        "llm_unavailable": "LLM no disponible: {error}",
        "no_new_observations": "{agent}: sin observaciones nuevas.",
        "step_limit": "{agent}: alcancé el límite de {steps} pasos sin conclusión.",
        "needs_llm": "{agent} necesita un LLM para responder preguntas.",
        # ---- progress
        "progress_start": "{agent}: iniciando.",
        "progress_rules": "Reglas aplicadas: {total} hallazgos ({critical} críticos, {warnings} avisos).",
        "progress_llm_step": "Consultando al modelo de lenguaje (paso {step})…",
        "progress_llm_tool": "El modelo usa la herramienta «{tool}».",
        "progress_llm_tool_error": "La herramienta «{tool}» devolvió un error; el modelo puede corregirse.",
        "progress_llm_done": "El modelo terminó su interpretación.",
        "progress_llm_failed": "El modelo de lenguaje no respondió; se muestran sólo las reglas.",
        "progress_next": "Avisando a otros agentes: {event}.",
        "progress_done": "{agent}: listo.",
        "progress_profile": "Perfil listo: {rows} filas, {columns} columnas, {nulls} nulos y {duplicates} duplicados.",
        "progress_clustering_matrix": "Matriz preparada: {rows} filas y {features} variables.",
        "progress_kmeans": "K-Means evaluado de k=2 a k={k_max} (mejor silhouette {score}).",
        "progress_ward": "HAC-ward evaluado (mejor silhouette {score}).",
        "progress_partition": "Partición elegida: {algorithm} con k={k}.",
        "progress_benchmark_start": "Comparando {candidates} candidatos con validación cruzada de {folds} pliegues; puede tardar.",
        "progress_benchmark_done": "Comparación lista: gana {algorithm} ({metric} de validación {score}).",
        # ---- EDA findings
        "eda_duplicates": "Hay {count} filas duplicadas.",
        "eda_nulls": "'{column}' tiene {percent}% de nulos.",
        "eda_constant": "'{column}' es constante y no aporta información.",
        "eda_identifier": "'{column}' parece un identificador; conviene excluirla.",
        "eda_no_target": "No hay variable objetivo; defínala para habilitar el modelado.",
        "eda_leakage": "'{column}' tiene r={r} con el target: posible fuga de información.",
        "eda_imbalance": "Target desbalanceado: la clase minoritaria es {share:.1%}.",
        # ---- clustering findings
        "clustering_too_few_rows": "El agrupamiento requiere al menos tres filas.",
        "clustering_recommendation": "Mejor partición: {algorithm} con k={k} (silhouette {silhouette}).",
        "clustering_weak": "Silhouette {silhouette}: no hay una estructura de grupos sustancial; interprete los clusters con cautela.",
        "clustering_small": "El cluster {cluster} sólo tiene {size} filas; puede ser un grupo de valores atípicos.",
        "clustering_unscaled": "Sin estandarizar, las variables de mayor escala dominan las distancias.",
        "clustering_dropped": "Se descartaron por no variar: {columns}.",
        "clustering_identifier": "'{column}' parece un identificador y distorsiona las distancias; conviene excluirla.",
        "clustering_nulls": "'{column}' tiene {percent:.1f}% de nulos imputados con la mediana.",
        # ---- supervised findings
        "supervised_no_target": "Defina la variable objetivo antes de entrenar modelos.",
        "supervised_failed": "{count} candidatos fallaron; revise la tabla comparativa.",
        "supervised_few_rows": "Sólo hay {ratio:.1f} filas de entrenamiento por variable; el modelo puede sobreajustar o ser inestable.",
        "regression_recommendation": "Mejor regresor: {algorithm} (RMSE de validación {rmse}, R² de prueba {r2}).",
        "regression_no_better_than_mean": "R² de prueba {r2}: el modelo no supera a predecir siempre la media.",
        "regression_weak": "R² de prueba {r2}: el modelo explica poca variación del target.",
        "regression_overfitting": "R² de entrenamiento {train} frente a {test} en prueba: posible sobreajuste.",
        "regression_unstable": "El RMSE varía mucho entre pliegues ({std} sobre {rmse}); la estimación es inestable.",
        "regression_test_gap": "El RMSE de prueba ({test}) se aleja del de validación ({validation}).",
        "classification_recommendation": "Mejor clasificador: {algorithm} (F1 macro de validación {validation}, F1 macro de prueba {test}).",
        "classification_no_better_than_majority": "Accuracy de prueba {accuracy}: no supera a predecir siempre la clase mayoritaria ({majority}).",
        "classification_overfitting": "F1 macro de entrenamiento {train} frente a {test} en prueba: posible sobreajuste.",
        "classification_misleading_accuracy": "Accuracy {accuracy} muy por encima del F1 macro {f1}: el desbalance infla la accuracy.",
        "classification_weak_class": "La clase '{label}' sólo se detecta en {recall:.0%} de los casos.",
        "classification_unstable": "El F1 macro varía mucho entre pliegues (desviación {std}); la estimación es inestable.",
    },
    "en": {
        # ---- agent names
        "agent_eda": "EDA agent",
        "agent_clustering": "Clustering agent",
        "agent_regression": "Regression agent",
        "agent_classification": "Classification agent",
        # ---- base agent
        "llm_unavailable": "LLM unavailable: {error}",
        "no_new_observations": "{agent}: no new observations.",
        "step_limit": "{agent}: reached the limit of {steps} steps without a conclusion.",
        "needs_llm": "{agent} needs an LLM to answer questions.",
        # ---- progress
        "progress_start": "{agent}: starting.",
        "progress_rules": "Rules applied: {total} findings ({critical} critical, {warnings} warnings).",
        "progress_llm_step": "Asking the language model (step {step})…",
        "progress_llm_tool": "The model is using the tool “{tool}”.",
        "progress_llm_tool_error": "The tool “{tool}” returned an error; the model can correct itself.",
        "progress_llm_done": "The model finished its interpretation.",
        "progress_llm_failed": "The language model did not respond; only the rules are shown.",
        "progress_next": "Notifying other agents: {event}.",
        "progress_done": "{agent}: done.",
        "progress_profile": "Profile ready: {rows} rows, {columns} columns, {nulls} nulls and {duplicates} duplicates.",
        "progress_clustering_matrix": "Matrix prepared: {rows} rows and {features} variables.",
        "progress_kmeans": "K-Means evaluated from k=2 to k={k_max} (best silhouette {score}).",
        "progress_ward": "HAC-ward evaluated (best silhouette {score}).",
        "progress_partition": "Chosen partition: {algorithm} with k={k}.",
        "progress_benchmark_start": "Comparing {candidates} candidates with {folds}-fold cross-validation; this may take a while.",
        "progress_benchmark_done": "Comparison ready: {algorithm} wins (validation {metric} {score}).",
        # ---- EDA findings
        "eda_duplicates": "There are {count} duplicated rows.",
        "eda_nulls": "'{column}' has {percent}% nulls.",
        "eda_constant": "'{column}' is constant and adds no information.",
        "eda_identifier": "'{column}' looks like an identifier; consider excluding it.",
        "eda_no_target": "There is no target variable; set one to enable modeling.",
        "eda_leakage": "'{column}' has r={r} with the target: possible data leakage.",
        "eda_imbalance": "Imbalanced target: the minority class is {share:.1%}.",
        # ---- clustering findings
        "clustering_too_few_rows": "Clustering requires at least three rows.",
        "clustering_recommendation": "Best partition: {algorithm} with k={k} (silhouette {silhouette}).",
        "clustering_weak": "Silhouette {silhouette}: there is no substantial group structure; interpret the clusters with caution.",
        "clustering_small": "Cluster {cluster} only has {size} rows; it may be a group of outliers.",
        "clustering_unscaled": "Without standardization, the variables with larger scales dominate the distances.",
        "clustering_dropped": "Dropped because they do not vary: {columns}.",
        "clustering_identifier": "'{column}' looks like an identifier and distorts the distances; consider excluding it.",
        "clustering_nulls": "'{column}' has {percent:.1f}% nulls imputed with the median.",
        # ---- supervised findings
        "supervised_no_target": "Set the target variable before training models.",
        "supervised_failed": "{count} candidates failed; check the comparison table.",
        "supervised_few_rows": "There are only {ratio:.1f} training rows per variable; the model may overfit or be unstable.",
        "regression_recommendation": "Best regressor: {algorithm} (validation RMSE {rmse}, test R² {r2}).",
        "regression_no_better_than_mean": "Test R² {r2}: the model is no better than always predicting the mean.",
        "regression_weak": "Test R² {r2}: the model explains little of the target's variation.",
        "regression_overfitting": "Training R² {train} versus {test} on test: possible overfitting.",
        "regression_unstable": "The RMSE varies a lot between folds ({std} over {rmse}); the estimate is unstable.",
        "regression_test_gap": "The test RMSE ({test}) is far from the validation RMSE ({validation}).",
        "classification_recommendation": "Best classifier: {algorithm} (validation F1 macro {validation}, test F1 macro {test}).",
        "classification_no_better_than_majority": "Test accuracy {accuracy}: no better than always predicting the majority class ({majority}).",
        "classification_overfitting": "Training F1 macro {train} versus {test} on test: possible overfitting.",
        "classification_misleading_accuracy": "Accuracy {accuracy} well above the F1 macro {f1}: the imbalance inflates the accuracy.",
        "classification_weak_class": "Class '{label}' is only detected in {recall:.0%} of the cases.",
        "classification_unstable": "The F1 macro varies a lot between folds (std {std}); the estimate is unstable.",
    },
}


def normalize_language(language: str | None) -> str:
    return language if language in MESSAGES else DEFAULT_LANGUAGE


def message(language: str | None, key: str, **values) -> str:
    """Text for ``key`` in ``language``, falling back to Spanish."""
    template = MESSAGES[normalize_language(language)].get(key) or MESSAGES[DEFAULT_LANGUAGE][key]
    return template.format(**values)


def language_instruction(language: str | None) -> str:
    """Line added to every system prompt so the LLM answers in the user's language."""
    name = LANGUAGE_NAMES[normalize_language(language)]
    return (
        f"Always answer in {name}, the language of the user's interface. If the user "
        "writes to you in another language, answer in the language the user wrote in."
    )
