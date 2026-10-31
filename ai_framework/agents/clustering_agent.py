"""Agente de agrupamiento construido sobre :class:`Agent`."""

from __future__ import annotations

from ai_framework.utils.results import Hallazgo, SugerenciaClustering
from ai_framework.utils.agents.agent_config import UmbralesClustering
from ai_framework.utils.unsupervised import evaluar_hac, evaluar_kmeans
from ai_framework.agents.agent import Agent
from ai_framework.agents.events import CLUSTERING_LISTO, CLUSTERING_SOLICITADO, Event
from ai_framework.agents.tools.clustering_tools import (
    HERRAMIENTAS_AGRUPAMIENTO,
    crear_cluster,
    resumir_resultado,
)
from ai_framework.agents.tools.eda_tools import parece_identificador


class ClusteringAgent(Agent):
    """Busca la mejor partición al solicitarse, avisa riesgos y describe los grupos."""

    nombre = "clustering"
    rol = (
        "Eres el agente de agrupamiento de un framework de aprendizaje automático. "
        "Usa sólo las herramientas para obtener cifras; no inventes valores. Explica "
        "en español, en pocos párrafos, cuántos grupos conviene usar, qué tan sólida "
        "es la estructura según silhouette y qué distingue a cada grupo. Las variables "
        "distintivas están en desviaciones estándar respecto de la media. Los clusters "
        "son una descripción de los datos, no categorías verdaderas: no les atribuyas "
        "causas ni significado que las cifras no respalden."
    )
    herramientas = HERRAMIENTAS_AGRUPAMIENTO
    escucha = frozenset({CLUSTERING_SOLICITADO})

    def __init__(
        self,
        pizarra,
        llm=None,
        *,
        features=None,
        incluir_categoricas: bool = False,
        estandarizar: bool = True,
        umbrales: UmbralesClustering | None = None,
    ) -> None:
        super().__init__(pizarra, llm)
        self.features = features
        self.incluir_categoricas = incluir_categoricas
        self.estandarizar = estandarizar
        self.umbrales = umbrales or UmbralesClustering()

    def preparar(self, evento: Event) -> None:
        """Compara K-Means y HAC-ward; otros métodos suelen premiar grupos degenerados."""
        cluster = crear_cluster(
            self.pizarra,
            features=self.features,
            incluir_categoricas=self.incluir_categoricas,
            estandarizar=self.estandarizar,
        )
        matriz = cluster.preparar_matriz().matriz
        filas = len(matriz)
        if filas < 3:
            raise ValueError("El agrupamiento requiere al menos tres filas.")
        k_max = min(self.umbrales.k_max, filas - 1)
        kmeans = evaluar_kmeans(matriz, 2, k_max)
        ward = evaluar_hac(matriz, 2, k_max, metodos=("ward",)).dropna(subset=["silhouette"])
        mejor_kmeans = kmeans.loc[kmeans["silhouette"].idxmax()]
        if not ward.empty and ward["silhouette"].max() > mejor_kmeans["silhouette"]:
            k = int(ward.loc[ward["silhouette"].idxmax(), "k"])
            resultado = cluster.HAC(k, metodo="ward")
        else:
            resultado = cluster.K_means(int(mejor_kmeans["k"]))
        self.pizarra.resultados.update(
            resultado_cluster=resultado,
            evaluacion_kmeans=kmeans,
            evaluacion_ward=ward,
            clustering=resumir_resultado(resultado),
        )

    def proponer(self, evento: Event) -> list[Hallazgo]:
        resultado = self.pizarra.resultados["resultado_cluster"]
        resumen = self.pizarra.resultados["clustering"]
        preparados = resultado.datos_preparados
        hallazgos: list[Hallazgo] = []

        def registrar(nivel, mensaje, **detalle):
            hallazgos.append(Hallazgo(self.nombre, nivel, mensaje, detalle))

        registrar("info",
                  f"Mejor partición: {resumen['algoritmo']} con k={resumen['n_clusters']} "
                  f"(silhouette {resumen['silhouette']}).", regla="recomendacion")
        if resumen["silhouette"] < self.umbrales.silhouette_debil:
            registrar("aviso",
                      f"Silhouette {resumen['silhouette']}: no hay una estructura de grupos "
                      "sustancial; interprete los clusters con cautela.",
                      regla="estructura_debil")
        for grupo in resumen["grupos"]:
            if grupo["proporcion"] < self.umbrales.cluster_pequeno:
                registrar("aviso",
                          f"El cluster {grupo['cluster']} sólo tiene {grupo['tamano']} filas; "
                          "puede ser un grupo de valores atípicos.",
                          regla="cluster_pequeno", cluster=grupo["cluster"])
        if not self.estandarizar and len(preparados.matriz.columns) > 1:
            registrar("aviso",
                      "Sin estandarizar, las variables de mayor escala dominan las distancias.",
                      regla="sin_escalar")
        if preparados.columnas_descartadas:
            registrar("info",
                      f"Se descartaron por no variar: {list(preparados.columnas_descartadas)}.",
                      regla="descartadas")
        for columna in preparados.columnas_origen:
            serie = self.pizarra.datos[columna]
            if parece_identificador(serie, self.umbrales.min_filas_identificador):
                registrar("aviso",
                          f"'{columna}' parece un identificador y distorsiona las distancias; "
                          "conviene excluirla.", regla="identificador", columna=columna)
            nulos = float(serie.isna().mean() * 100)
            if nulos > self.umbrales.nulos:
                registrar("aviso",
                          f"'{columna}' tiene {nulos:.1f}% de nulos imputados con la mediana.",
                          regla="nulos", columna=columna)
        return hallazgos

    def siguientes(self, evento: Event) -> list[Event]:
        resumen = self.pizarra.resultados["clustering"]
        return [
            Event(
                CLUSTERING_LISTO,
                {"algoritmo": resumen["algoritmo"], "n_clusters": resumen["n_clusters"]},
                origen=self.nombre,
            )
        ]

    def sugerir_agrupamiento(self) -> SugerenciaClustering | None:
        resumen = self.pizarra.resultados.get("clustering")
        if resumen is None:
            return None
        return SugerenciaClustering(
            resumen["algoritmo"],
            resumen["n_clusters"],
            resumen["silhouette"],
            tuple(grupo["tamano"] for grupo in resumen["grupos"]),
        )
