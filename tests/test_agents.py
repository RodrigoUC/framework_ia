"""Agentes: bus, pizarra, herramientas, bucle con LLM falso, EDAAgent y CLI."""

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_framework.agents import (
    DATASET_CARGADO,
    EDA_LISTO,
    Agent,
    EDAAgent,
    EventBus,
    ErrorProveedor,
    Event,
    Blackboard,
    ProveedorFalso,
    ProveedorOpenAICompat,
    RespuestaLLM,
    crear_proveedor,
)
from ai_framework.agents.tools.eda_tools import tipo_tarea
from ai_framework.agents.tools.registry import ejecutar, herramienta
from ai_framework.agents.llm import LlamadaHerramienta
from ai_framework.data.eda import EDA
from ai_framework.utils.results import Hallazgo, SugerenciaModelado
from ai_framework.ui.models.agents_state import (
    limpiar_resultados_agentes,
    sincronizar_contexto_agentes,
)
from ai_framework.utils.agents.agent_config import PROVEEDOR_SIN_LLM, UmbralesEDA
from scripts.ejecutar_agentes import main


@pytest.fixture
def datos_regresion():
    rng = np.random.default_rng(7)
    x, ruido = rng.normal(size=(2, 60))
    return pd.DataFrame(
        {
            "id": np.arange(60),
            "x": x,
            "ruido": ruido,
            "constante": 1.0,
            "faltante": np.where(np.arange(60) % 2 == 0, np.nan, ruido),
            "fuga": 2 * x + 1,
            "y": 2 * x + 1 + 0.001 * ruido,
        }
    )


@pytest.fixture
def datos_desbalanceados():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 100))
    return pd.DataFrame({"a": a, "b": b, "clase": [1] * 5 + [0] * 95})


def sistema(datos, target, llm=None):
    pizarra = Blackboard(datos, target=target)
    bus = EventBus()
    agente = EDAAgent(pizarra, llm).conectar(bus)
    recibidos = []
    bus.suscribir(EDA_LISTO, recibidos.append)
    return pizarra, bus, agente, recibidos


def llamada(nombre, id_="c1", **argumentos):
    return RespuestaLLM(llamadas=(LlamadaHerramienta(id_, nombre, argumentos),))


def cliente_falso(mensaje=None, error=None):
    pedidos = []

    def crear(**parametros):
        pedidos.append(parametros)
        if error is not None:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=mensaje)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=crear))), pedidos


# ---------------------------------------------------------------- infraestructura
def test_bus_procesa_eventos_anidados_en_orden_fifo():
    bus = EventBus()
    orden = []

    def primero(evento):
        orden.append("a")
        bus.emitir(Event("b"))
        orden.append("a-fin")

    bus.suscribir("a", primero)
    bus.suscribir("b", lambda evento: orden.append("b"))
    bus.emitir(Event("a"))

    assert orden == ["a", "a-fin", "b"]
    assert [e.tipo for e in bus.historial] == ["a", "b"]


def test_pizarra_es_un_eda_con_estado_compartido(datos_desbalanceados):
    pizarra = Blackboard(datos_desbalanceados, target="clase")

    assert isinstance(pizarra, EDA)
    assert pizarra.resumen_calidad()["filas"] == 100
    with pytest.raises(KeyError):
        Blackboard(datos_desbalanceados, target="z")


def test_pizarra_no_duplica_hallazgos_ni_envia_filas_al_llm():
    pizarra = Blackboard(pd.DataFrame({"secreto": ["valor-privado-123"]}))
    hallazgo = Hallazgo("eda", "info", "x")

    assert pizarra.registrar(hallazgo) and not pizarra.registrar(hallazgo)
    resumen = pizarra.resumen_para_llm()
    assert "secreto" in resumen and "valor-privado-123" not in resumen


def test_agente_es_abstracto(datos_desbalanceados):
    with pytest.raises(TypeError):
        Agent(Blackboard(datos_desbalanceados))


# ---------------------------------------------------------------- herramientas
@pytest.fixture
def demo():
    @herramienta("Prueba.", n="Cantidad.")
    def demo(pizarra, nombre: str, n: int = 3, columnas: list[str] | None = None) -> dict:
        if n < 0:
            raise ValueError("n negativo")
        return {"nombre": nombre, "n": n}

    return demo


def test_esquema_excluye_pizarra_y_marca_requeridos(demo):
    parametros = demo.esquema()["function"]["parameters"]

    assert set(parametros["properties"]) == {"nombre", "n", "columnas"}
    assert parametros["required"] == ["nombre"]
    assert parametros["properties"]["n"] == {"type": "integer", "description": "Cantidad."}
    assert parametros["properties"]["columnas"] == {"type": "array", "items": {"type": "string"}}


@pytest.mark.parametrize(
    "nombre,argumentos",
    [
        ("otra", {}),
        ("demo", {"nombre": "a", "x": 1}),
        ("demo", {"__json_invalido__": "{"}),
        ("demo", {}),
        ("demo", {"nombre": "a", "n": -1}),
    ],
    ids=["desconocida", "argumento-extra", "json-invalido", "falta-requerido", "excepcion"],
)
def test_ejecutar_devuelve_errores_en_lugar_de_lanzar(demo, nombre, argumentos):
    assert "error" in ejecutar(LlamadaHerramienta("1", nombre, argumentos), [demo], None)


def test_ejecutar_llama_la_herramienta(demo):
    assert ejecutar(LlamadaHerramienta("1", "demo", {"nombre": "a"}), [demo], None) == {
        "nombre": "a", "n": 3,
    }


@pytest.mark.parametrize(
    "serie,tarea",
    [(pd.Series(range(50)), "regresion"), (pd.Series([0, 1] * 25), "clasificacion"),
     (pd.Series(["a", "b"] * 25), "clasificacion")],
)
def test_tipo_tarea(serie, tarea):
    assert tipo_tarea(serie) == tarea


# ---------------------------------------------------------------- EDAAgent sin LLM
def test_reglas_proactivas_detectan_riesgos_y_habilitan_regresion(datos_regresion):
    pizarra, bus, agente, recibidos = sistema(datos_regresion, "y")
    bus.emitir(Event(DATASET_CARGADO))

    reglas = {h.detalle.get("columna"): h.detalle.get("regla") for h in pizarra.hallazgos}
    assert reglas == {
        "id": "identificador", "constante": "constante", "faltante": "nulos",
        "x": "fuga", "fuga": "fuga",
    }
    assert recibidos[0].datos == {"target": "y", "tarea": "regresion"}
    assert recibidos[0].origen == "eda"
    assert agente.sugerir_configuracion() == SugerenciaModelado(
        "regresion", "y", ("ruido", "faltante"), ("constante", "fuga", "id", "x")
    )


def test_umbrales_configurables(datos_regresion):
    pizarra = Blackboard(datos_regresion, target="y")
    EDAAgent(pizarra, umbrales=UmbralesEDA(nulos=60.0)).conectar(bus := EventBus())
    bus.emitir(Event(DATASET_CARGADO))

    assert "nulos" not in {h.detalle.get("regla") for h in pizarra.hallazgos}


def test_detecta_desbalance_en_clasificacion(datos_desbalanceados):
    pizarra, bus, _, recibidos = sistema(datos_desbalanceados, "clase")
    bus.emitir(Event(DATASET_CARGADO))

    assert recibidos[0].datos["tarea"] == "clasificacion"
    assert any(h.detalle.get("regla") == "desbalance" for h in pizarra.hallazgos)


def test_sin_target_no_emite_eda_listo(datos_desbalanceados):
    pizarra, bus, agente, recibidos = sistema(datos_desbalanceados, None)
    bus.emitir(Event(DATASET_CARGADO))

    assert recibidos == []
    assert agente.sugerir_configuracion() is None
    assert any("objetivo" in h.mensaje for h in pizarra.hallazgos)


def test_preguntar_sin_llm_falla_explicitamente(datos_desbalanceados):
    _, _, agente, _ = sistema(datos_desbalanceados, "clase")
    with pytest.raises(RuntimeError):
        agente.preguntar("¿Qué ves?")


# ---------------------------------------------------------------- EDAAgent con LLM falso
def test_bucle_ejecuta_herramienta_y_devuelve_interpretacion(datos_regresion):
    llm = ProveedorFalso([llamada("correlaciones", umbral=0.9), RespuestaLLM(texto="Hay fuga.")])
    pizarra, bus, _, recibidos = sistema(datos_regresion, "y", llm)
    bus.emitir(Event(DATASET_CARGADO))

    assert pizarra.mensajes[-1] == ("eda", "Hay fuga.")
    assert len(recibidos) == 1
    primera, segunda = llm.llamadas
    assert {h["function"]["name"] for h in primera["herramientas"]} == {
        "perfil_dataset", "correlaciones", "valores_atipicos", "frecuencias_columna",
    }
    assert "perfil" in primera["mensajes"][1]["content"]
    asistente, respuesta_tool = segunda["mensajes"][-2:]
    assert asistente["tool_calls"][0]["function"]["name"] == "correlaciones"
    assert (respuesta_tool["role"], respuesta_tool["tool_call_id"]) == ("tool", "c1")
    assert {"a": "fuga", "b": "y", "r": 1.0} in json.loads(respuesta_tool["content"])["pares"]


def test_error_de_herramienta_vuelve_al_llm(datos_desbalanceados):
    llm = ProveedorFalso([llamada("frecuencias_columna", columna="no_existe"), RespuestaLLM(texto="ok")])
    _, bus, _, _ = sistema(datos_desbalanceados, "clase", llm)
    bus.emitir(Event(DATASET_CARGADO))

    assert "KeyError" in json.loads(llm.llamadas[1]["mensajes"][-1]["content"])["error"]


def test_falla_del_proveedor_no_detiene_el_flujo(datos_desbalanceados):
    pizarra, bus, _, recibidos = sistema(datos_desbalanceados, "clase", ProveedorFalso([]))
    bus.emitir(Event(DATASET_CARGADO))

    assert any("LLM no disponible" in h.mensaje for h in pizarra.hallazgos)
    assert len(recibidos) == 1
    assert "desbalanceado" in pizarra.mensajes[-1][1]


def test_limite_de_pasos(datos_desbalanceados):
    llm = ProveedorFalso([llamada("perfil_dataset", id_=str(i)) for i in range(10)])
    _, _, agente, _ = sistema(datos_desbalanceados, "clase", llm)
    agente.max_pasos = 2

    assert "límite" in agente.preguntar("Analiza")
    assert len(llm.llamadas) == 2


def test_preguntar_registra_conversacion(datos_desbalanceados):
    llm = ProveedorFalso([RespuestaLLM(texto="Respuesta")])
    pizarra, _, agente, _ = sistema(datos_desbalanceados, "clase", llm)

    assert agente.preguntar("¿Hay nulos?") == "Respuesta"
    assert pizarra.mensajes == [("usuario", "¿Hay nulos?"), ("eda", "Respuesta")]
    assert llm.llamadas[0]["mensajes"][1]["content"].startswith("¿Hay nulos?")


def test_consola_muestra_prompt_herramientas_y_respuesta(capsys, datos_desbalanceados):
    llm = ProveedorFalso([llamada("correlaciones", umbral=0.9), RespuestaLLM(texto="Respuesta final")])
    _, _, agente, _ = sistema(datos_desbalanceados, "clase", llm)
    agente.preguntar("¿Hay fuga?")

    salida = capsys.readouterr().out
    assert agente.rol in salida
    assert 'correlaciones({"umbral": 0.9})' in salida
    assert '"pares"' in salida
    assert "Respuesta final" in salida
    # El prompt inicial no se repite aunque el historial se reenvíe en cada paso.
    assert salida.count("¿Hay fuga?") == 1


def test_consola_muestra_el_error_del_proveedor(capsys, datos_desbalanceados):
    _, _, agente, _ = sistema(datos_desbalanceados, "clase", ProveedorFalso([]))
    with pytest.raises(RuntimeError):
        agente.preguntar("Hola")

    assert "ERROR paso 1" in capsys.readouterr().out


def test_consola_sin_soporte_de_caracteres_no_tumba_al_agente(monkeypatch, datos_desbalanceados):
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    llm = ProveedorFalso([RespuestaLLM(texto="Listo \U0001F600")])
    _, _, agente, _ = sistema(datos_desbalanceados, "clase", llm)

    assert agente.preguntar("Hola") == "Listo \U0001F600"


# ---------------------------------------------------------------- proveedor
def test_proveedor_normaliza_llamadas_y_json_invalido():
    mensaje = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(id="a", function=SimpleNamespace(name="f", arguments='{"x": 1}')),
            SimpleNamespace(id="b", function=SimpleNamespace(name="g", arguments="{roto")),
        ],
    )
    cliente, pedidos = cliente_falso(mensaje)
    respuesta = ProveedorOpenAICompat("groq", "modelo-x", cliente=cliente).completar(
        [{"role": "user", "content": "hola"}], [{"type": "function"}]
    )

    assert respuesta.llamadas[0] == LlamadaHerramienta("a", "f", {"x": 1})
    assert respuesta.llamadas[1].argumentos == {"__json_invalido__": "{roto"}
    assert pedidos[0]["model"] == "modelo-x" and "tools" in pedidos[0]


def test_thought_signature_vuelve_a_gemini_con_el_sdk_real(datos_desbalanceados):
    """Recorre el SDK openai real contra un servidor en 127.0.0.1, sin salir a la red."""
    openai = pytest.importorskip("openai")
    firma = {"google": {"thought_signature": "firma-opaca"}}
    pedidos = []

    class Servidor(BaseHTTPRequestHandler):
        def do_POST(self):
            pedidos.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            mensaje = {"role": "assistant", "content": "Listo"}
            if len(pedidos) == 1:
                mensaje = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "c1", "type": "function", "extra_content": firma,
                        "function": {"name": "correlaciones", "arguments": '{"umbral": 0.5}'},
                    }],
                }
            cuerpo = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": mensaje}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def log_message(self, *argumentos):
            pass

    servidor = ThreadingHTTPServer(("127.0.0.1", 0), Servidor)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    try:
        cliente = openai.OpenAI(
            base_url=f"http://127.0.0.1:{servidor.server_port}/v1", api_key="k", max_retries=0
        )
        llm = ProveedorOpenAICompat("gemini", "m", cliente=cliente)
        _, _, agente, _ = sistema(datos_desbalanceados, "clase", llm)

        assert agente.preguntar("Analiza") == "Listo"
    finally:
        servidor.shutdown()
        servidor.server_close()
    asistente = next(m for m in pedidos[1]["messages"] if m["role"] == "assistant")
    assert asistente["tool_calls"][0]["extra_content"] == firma


def test_proveedor_sin_herramientas_no_envia_tools():
    cliente, pedidos = cliente_falso(SimpleNamespace(content="hola", tool_calls=None))

    assert ProveedorOpenAICompat("ollama", "m", cliente=cliente).completar([], []) == RespuestaLLM(texto="hola")
    assert "tools" not in pedidos[0]


def test_error_http_incluye_motivo_y_pista():
    class ErrorHTTP(Exception):
        status_code = 403
        body = [{"error": {"message": "Generative Language API has not been used"}}]

    cliente, _ = cliente_falso(error=ErrorHTTP("Error code: 403"))
    with pytest.raises(ErrorProveedor) as contexto:
        ProveedorOpenAICompat("gemini", "m", cliente=cliente).completar([], [])

    assert contexto.value.estado == 403
    assert "Generative Language API has not been used" in str(contexto.value)
    assert "habilite la API" in str(contexto.value)


def test_proveedor_valida_configuracion(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError):
        ProveedorOpenAICompat("desconocido", "m", cliente=object())
    with pytest.raises(ValueError):
        ProveedorOpenAICompat("gemini", "m")


def test_modelo_por_defecto_es_el_sugerido(monkeypatch):
    monkeypatch.delenv("AGENTES_MODELO", raising=False)
    assert ProveedorOpenAICompat("groq", cliente=object()).modelo == "llama-3.3-70b-versatile"


def test_crear_proveedor_sin_llm():
    assert crear_proveedor(PROVEEDOR_SIN_LLM) is None


# ---------------------------------------------------------------- estado de interfaz
def test_estado_agentes_se_invalida_al_cambiar_datos(datos_desbalanceados):
    estado = {"dataset_identidad": "a"}
    sincronizar_contexto_agentes(estado, datos_desbalanceados)
    estado["agentes_sesion"] = object()

    sincronizar_contexto_agentes(estado, datos_desbalanceados)
    assert "agentes_sesion" in estado
    sincronizar_contexto_agentes(estado, datos_desbalanceados.iloc[:50])
    assert "agentes_sesion" not in estado

    estado["agentes_sesion"] = object()
    limpiar_resultados_agentes(estado)
    assert not {"agentes_sesion", "agentes_contexto"} & set(estado)


# ---------------------------------------------------------------- CLI
def test_cli_sin_llm_imprime_hallazgos_y_sugerencia(tmp_path, capsys, datos_regresion):
    ruta = tmp_path / "datos.csv"
    datos_regresion.to_csv(ruta, index=False)

    assert main([str(ruta), "--target", "y"]) == 0
    salida = capsys.readouterr().out
    assert "EDA listo: tarea = regresion" in salida
    assert "posible fuga" in salida
    assert "Sugerencia: regresion sobre y" in salida


def test_cli_interactivo_exige_llm(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.csv"), "--interactivo"])
