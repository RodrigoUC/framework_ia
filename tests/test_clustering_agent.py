"""ClusteringAgent: búsqueda determinista, reglas, herramientas, vista y CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_framework.agents import (
    CLUSTERING_LISTO,
    CLUSTERING_SOLICITADO,
    ClusteringAgent,
    EventBus,
    Event,
    Blackboard,
    ProveedorFalso,
    RespuestaLLM,
)
from ai_framework.agents.tools.clustering_tools import (
    crear_cluster,
    ejecutar_agrupamiento,
    evaluar_jerarquico,
    evaluar_k,
    resumir_resultado,
)
from ai_framework.agents.tools.registry import ejecutar
from ai_framework.agents.llm import LlamadaHerramienta
from ai_framework.utils.results import SugerenciaClustering
from ai_framework.ui.models.agents_state import (
    limpiar_resultados_agentes,
    sincronizar_contexto_agentes,
    sincronizar_contexto_clustering,
)
from ai_framework.utils.agents.agent_config import PROVEEDOR_SIN_LLM, UmbralesClustering
from scripts.ejecutar_agentes import main

ENTRYPOINT = Path(__file__).with_name("streamlit_app_entry.py")


def grupos(n_por_grupo=30, centros=((0, 0), (10, 10), (-10, 10))):
    rng = np.random.default_rng(11)
    filas = [rng.normal(centro, 0.4, size=(n_por_grupo, 2)) for centro in centros]
    return pd.DataFrame(np.vstack(filas), columns=["a", "b"])


def sistema(datos, llm=None, **opciones):
    pizarra = Blackboard(datos)
    bus = EventBus()
    agente = ClusteringAgent(pizarra, llm, **opciones).conectar(bus)
    recibidos = []
    bus.suscribir(CLUSTERING_LISTO, recibidos.append)
    return pizarra, bus, agente, recibidos


def reglas(pizarra):
    return {h.detalle.get("regla") for h in pizarra.hallazgos}


def test_encuentra_los_grupos_reales_sin_llm():
    pizarra, bus, agente, recibidos = sistema(grupos())
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    sugerencia = agente.sugerir_agrupamiento()
    assert isinstance(sugerencia, SugerenciaClustering)
    assert sugerencia.n_clusters == 3
    assert sugerencia.silhouette > 0.8
    assert sugerencia.tamanos == (30, 30, 30)
    assert reglas(pizarra) == {"recomendacion"}
    assert recibidos[0].datos["n_clusters"] == 3
    assert recibidos[0].origen == "clustering"


def test_sin_estructura_avisa_que_es_debil():
    rng = np.random.default_rng(1)
    pizarra, bus, _, _ = sistema(pd.DataFrame(rng.uniform(size=(150, 6))))
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    assert "estructura_debil" in reglas(pizarra)


def test_avisa_cluster_pequeno():
    datos = pd.concat([grupos(100, ((0, 0), (10, 10))), pd.DataFrame({"a": [60.0] * 3, "b": [60.0] * 3})])
    pizarra, bus, agente, _ = sistema(datos.reset_index(drop=True))
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    assert "cluster_pequeno" in reglas(pizarra)
    assert min(agente.sugerir_agrupamiento().tamanos) == 3


def test_avisa_riesgos_de_las_variables():
    datos = grupos().assign(
        codigo=lambda d: np.arange(len(d)),
        constante=1.0,
        incompleta=lambda d: d["a"].where(d.index % 2 == 0),
    )
    pizarra, bus, _, _ = sistema(datos, estandarizar=False)
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    assert {"identificador", "descartadas", "nulos", "sin_escalar"} <= reglas(pizarra)


def test_umbrales_configurables():
    pizarra, bus, agente, _ = sistema(grupos(), umbrales=UmbralesClustering(k_max=2))
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    assert agente.sugerir_agrupamiento().n_clusters == 2
    assert pizarra.resultados["evaluacion_kmeans"]["k"].max() == 2


def test_columnas_categoricas_sin_codificar_fallan_antes_de_llamar_al_llm():
    llm = ProveedorFalso([])
    datos = grupos().assign(tipo=lambda d: np.where(d.index % 2 == 0, "x", "y"))
    _, bus, _, _ = sistema(datos, llm, features=["a", "tipo"])

    with pytest.raises(TypeError):
        bus.emitir(Event(CLUSTERING_SOLICITADO))
    assert llm.llamadas == []


def test_resumen_al_llm_no_expone_filas_ni_grupos_diminutos():
    datos = grupos(20, ((0, 0), (10, 10))).assign(a=lambda d: d["a"] + 0.123456789)
    datos.loc[0, ["a", "b"]] = [987654.0, 987654.0]
    pizarra, bus, _, _ = sistema(datos)
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    contexto = pizarra.resumen_para_llm()
    assert "987654" not in contexto
    assert "0.123456789" not in contexto
    assert "resultado_cluster" not in contexto
    aislado = next(g for g in json.loads(contexto)["clustering"]["grupos"] if g["tamano"] == 1)
    assert "distintivas" not in aislado


def test_llm_interpreta_con_contexto_y_herramientas():
    llm = ProveedorFalso([
        RespuestaLLM(llamadas=(LlamadaHerramienta("c1", "ejecutar_agrupamiento", {
            "algoritmo": "K-Medoids", "n_clusters": 3,
        }),)),
        RespuestaLLM(texto="Hay tres grupos."),
    ])
    pizarra, bus, _, _ = sistema(grupos(), llm)
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    assert pizarra.mensajes[-1] == ("clustering", "Hay tres grupos.")
    primera, segunda = llm.llamadas
    assert '"clustering"' in primera["mensajes"][1]["content"]
    assert {h["function"]["name"] for h in primera["herramientas"]} == {
        "evaluar_k", "evaluar_jerarquico", "ejecutar_agrupamiento",
    }
    respuesta = json.loads(segunda["mensajes"][-1]["content"])
    assert respuesta["algoritmo"] == "K-Medoids"
    # La exploración del LLM no reemplaza la recomendación determinista.
    assert pizarra.resultados["clustering"]["algoritmo"] != "K-Medoids"


def test_falla_del_llm_conserva_la_recomendacion():
    pizarra, bus, agente, recibidos = sistema(grupos(), ProveedorFalso([]))
    bus.emitir(Event(CLUSTERING_SOLICITADO))

    assert any("LLM no disponible" in h.mensaje for h in pizarra.hallazgos)
    assert agente.sugerir_agrupamiento().n_clusters == 3
    assert len(recibidos) == 1


def test_herramientas_de_agrupamiento():
    pizarra = Blackboard(grupos())
    crear_cluster(pizarra)

    assert [f["k"] for f in evaluar_k.funcion(pizarra, 2, 4)["filas"]] == [2, 3, 4]
    filas = evaluar_jerarquico.funcion(pizarra, 2, 3, ["ward", "single"])["filas"]
    assert {f["metodo"] for f in filas} == {"ward", "single"}
    hac = ejecutar_agrupamiento.funcion(pizarra, "HAC", 3, "average")
    assert hac["algoritmo"] == "HAC-average" and len(hac["grupos"]) == 3
    assert set(hac["grupos"][0]["distintivas"]) == {"a", "b"}


@pytest.mark.parametrize(
    "argumentos",
    [{"algoritmo": "ACP", "n_clusters": 2}, {"algoritmo": "HAC", "n_clusters": 2, "metodo_hac": "otro"},
     {"algoritmo": "K-Means", "n_clusters": 100000}],
    ids=["no-es-clustering", "metodo-invalido", "k-excesivo"],
)
def test_herramienta_devuelve_errores_al_llm(argumentos):
    pizarra = Blackboard(grupos())
    crear_cluster(pizarra)
    llamada = LlamadaHerramienta("1", "ejecutar_agrupamiento", argumentos)

    assert "error" in ejecutar(llamada, [ejecutar_agrupamiento], pizarra)


def test_resumen_mide_distintivas_en_desviaciones():
    cluster = crear_cluster(Blackboard(grupos()))
    resumen = resumir_resultado(cluster.K_means(3))

    assert sum(g["tamano"] for g in resumen["grupos"]) == 90
    assert all(abs(v) < 5 for g in resumen["grupos"] for v in g["distintivas"].values())


def test_estados_de_agentes_no_se_pisan():
    datos = grupos()
    configuracion = {"features": ["a", "b"], "incluir_categoricas": False, "estandarizar": True}
    estado = {"dataset_identidad": "x"}
    sincronizar_contexto_clustering(estado, datos, configuracion)
    estado["agente_clustering_sesion"] = object()

    # Abrir la vista del agente EDA no debe borrar la sesión de clustering.
    sincronizar_contexto_agentes(estado, datos)
    assert "agente_clustering_sesion" in estado
    sincronizar_contexto_clustering(estado, datos, configuracion)
    assert "agente_clustering_sesion" in estado
    sincronizar_contexto_clustering(estado, datos, {**configuracion, "estandarizar": False})
    assert "agente_clustering_sesion" not in estado

    estado["agente_clustering_sesion"] = object()
    limpiar_resultados_agentes(estado)
    assert "agente_clustering_sesion" not in estado


# ---------------------------------------------------------------- vista
def nueva_app(datos=None):
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["fuente_aplicada"] = (grupos() if datos is None else datos, "grupos.csv", "grupos-ui")
    app.run()
    app.sidebar.button(key="nav_clustering_agente").click().run()
    assert not app.exception
    return app


def iniciar(app, proveedor=PROVEEDOR_SIN_LLM):
    app.selectbox(key="agentes_proveedor").set_value(proveedor)
    next(b for b in app.button if b.label == "Iniciar agente").click()
    app.run()
    assert not app.exception
    return app


def test_navegar_no_agrupa_hasta_pulsar_el_boton():
    app = nueva_app()

    assert "agente_clustering_sesion" not in app.session_state
    assert any("Iniciar agente" in i.value for i in app.info)


def test_vista_muestra_recomendacion_sin_llm():
    app = iniciar(nueva_app())

    sesion = app.session_state["agente_clustering_sesion"]
    assert sesion["agente"].llm is None
    assert sesion["pizarra"].origen == "grupos.csv"
    metricas = {m.label: m.value for m in app.metric}
    assert metricas["Clusters"] == "3"
    assert float(metricas["Silhouette"]) > 0.8


def test_cambiar_variables_invalida_la_recomendacion():
    app = iniciar(nueva_app())
    assert "agente_clustering_sesion" in app.session_state

    app.multiselect(key="modelo_features").set_value(["a"]).run()
    assert not app.exception
    assert "agente_clustering_sesion" not in app.session_state


def test_vista_con_llm_falso_responde_preguntas():
    llm = ProveedorFalso([RespuestaLLM(texto="Tres grupos."), RespuestaLLM(texto="El grupo 0.")])
    with patch("ai_framework.ui.controllers.agents_controller.crear_proveedor", return_value=llm):
        app = iniciar(nueva_app(), proveedor="gemini")
        app.chat_input(key="agente_clustering_pregunta").set_value("¿Cuál es mayor?").run()

    textos = [m.markdown[0].value for m in app.chat_message]
    assert textos == ["Tres grupos.", "¿Cuál es mayor?", "El grupo 0."]


def test_error_de_proveedor_en_la_vista():
    with patch("ai_framework.ui.controllers.agents_controller.crear_proveedor", side_effect=ValueError("Defina GEMINI_API_KEY")):
        app = iniciar(nueva_app(), proveedor="gemini")

    assert any("GEMINI_API_KEY" in error.value for error in app.error)
    assert "agente_clustering_sesion" not in app.session_state


# ---------------------------------------------------------------- CLI
def test_cli_clustering(tmp_path, capsys):
    ruta = tmp_path / "grupos.csv"
    grupos().to_csv(ruta, index=False)

    assert main([str(ruta), "--agente", "clustering"]) == 0
    assert "k=3" in capsys.readouterr().out
