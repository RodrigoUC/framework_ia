"""Clustering agent built on :class:`Agent`."""

from __future__ import annotations

from ai_framework.utils.results import ClusteringSuggestion, Finding
from ai_framework.utils.agents.agent_config import ClusteringThresholds
from ai_framework.utils.unsupervised import evaluar_hac, evaluar_kmeans
from ai_framework.agents.agent import Agent
from ai_framework.agents.events import CLUSTERING_DONE, CLUSTERING_REQUESTED, Event
from ai_framework.agents.tools.clustering_tools import (
    CLUSTERING_TOOLS,
    create_clustering,
    summarize_result,
)
from ai_framework.agents.tools.eda_tools import looks_like_identifier


class ClusteringAgent(Agent):
    """Searches for the best partition when requested, flags risks and describes the groups."""

    name = "clustering"
    role = (
        "You are the clustering agent of a machine learning framework. Use only the "
        "tools to obtain figures; never invent values. Explain, in a few paragraphs, how "
        "many groups to use, how solid the structure is according to silhouette and what "
        "sets each group apart. Distinctive variables are measured in standard deviations "
        "from the mean. Clusters describe the data; they are not true categories, so do "
        "not attribute causes or meaning the figures do not support."
    )
    tools = CLUSTERING_TOOLS
    listens_to = frozenset({CLUSTERING_REQUESTED})

    def __init__(
        self,
        blackboard,
        llm=None,
        *,
        features=None,
        include_categorical: bool = False,
        standardize: bool = True,
        thresholds: ClusteringThresholds | None = None,
        **options,
    ) -> None:
        super().__init__(blackboard, llm, **options)
        self.features = features
        self.include_categorical = include_categorical
        self.standardize = standardize
        self.thresholds = thresholds or ClusteringThresholds()

    def prepare(self, event: Event) -> None:
        """Compares K-Means and HAC-ward; other linkages tend to reward degenerate groups."""
        model = create_clustering(
            self.blackboard,
            features=self.features,
            incluir_categoricas=self.include_categorical,
            estandarizar=self.standardize,
        )
        matrix = model.preparar_matriz().matriz
        rows = len(matrix)
        if rows < 3:
            raise ValueError(self.text("clustering_too_few_rows"))
        self.report("progress_clustering_matrix", rows=rows, features=matrix.shape[1])
        k_max = min(self.thresholds.k_max, rows - 1)
        kmeans = evaluar_kmeans(matrix, 2, k_max)
        best_kmeans = kmeans.loc[kmeans["silhouette"].idxmax()]
        self.report("progress_kmeans", k_max=k_max, score=round(float(best_kmeans["silhouette"]), 4))
        ward = evaluar_hac(matrix, 2, k_max, metodos=("ward",)).dropna(subset=["silhouette"])
        if not ward.empty:
            self.report("progress_ward", score=round(float(ward["silhouette"].max()), 4))
        if not ward.empty and ward["silhouette"].max() > best_kmeans["silhouette"]:
            k = int(ward.loc[ward["silhouette"].idxmax(), "k"])
            result = model.HAC(k, metodo="ward")
        else:
            result = model.K_means(int(best_kmeans["k"]))
        self.report("progress_partition", algorithm=result.algoritmo, k=result.etiquetas.nunique())
        self.blackboard.results.update(
            cluster_result=result,
            kmeans_evaluation=kmeans,
            ward_evaluation=ward,
            clustering=summarize_result(result),
        )

    def propose(self, event: Event) -> list[Finding]:
        result = self.blackboard.results["cluster_result"]
        summary = self.blackboard.results["clustering"]
        prepared = result.datos_preparados
        findings: list[Finding] = []

        def add(level, message, **detail):
            findings.append(Finding(self.name, level, message, detail))

        add("info", self.text("clustering_recommendation", algorithm=summary["algorithm"],
                              k=summary["n_clusters"], silhouette=summary["silhouette"]),
            rule="recommendation")
        if summary["silhouette"] < self.thresholds.weak_silhouette:
            add("warning", self.text("clustering_weak", silhouette=summary["silhouette"]),
                rule="weak_structure")
        for group in summary["groups"]:
            if group["share"] < self.thresholds.small_cluster:
                add("warning", self.text("clustering_small", cluster=group["cluster"], size=group["size"]),
                    rule="small_cluster", cluster=group["cluster"])
        if not self.standardize and len(prepared.matriz.columns) > 1:
            add("warning", self.text("clustering_unscaled"), rule="unscaled")
        if prepared.columnas_descartadas:
            add("info", self.text("clustering_dropped", columns=list(prepared.columnas_descartadas)),
                rule="dropped")
        for column in prepared.columnas_origen:
            series = self.blackboard.datos[column]
            if looks_like_identifier(series, self.thresholds.min_identifier_rows):
                add("warning", self.text("clustering_identifier", column=column),
                    rule="identifier", column=column)
            nulls = float(series.isna().mean() * 100)
            if nulls > self.thresholds.nulls:
                add("warning", self.text("clustering_nulls", column=column, percent=nulls),
                    rule="nulls", column=column)
        return findings

    def next_events(self, event: Event) -> list[Event]:
        summary = self.blackboard.results["clustering"]
        return [
            Event(
                CLUSTERING_DONE,
                {"algorithm": summary["algorithm"], "n_clusters": summary["n_clusters"]},
                source=self.name,
            )
        ]

    def suggest_clustering(self) -> ClusteringSuggestion | None:
        summary = self.blackboard.results.get("clustering")
        if summary is None:
            return None
        return ClusteringSuggestion(
            summary["algorithm"],
            summary["n_clusters"],
            summary["silhouette"],
            tuple(group["size"] for group in summary["groups"]),
        )
