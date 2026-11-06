# Agentes

Los agentes coordinan el framework con un modelo de lenguaje (LLM) gratuito.
El LLM decide qué analizar e interpreta; los cálculos siempre los hace el
framework. Nunca se genera ni ejecuta código, y al LLM sólo llega un resumen
del dataset, no sus filas.

## Arquitectura

Sigue las mismas capas que el resto del framework:

| Capa | Módulo | Rol |
| --- | --- | --- |
| Núcleo | `ai_framework/agents/blackboard.py` | `Blackboard(EDA)` (alias `Pizarra`): estado compartido; los agentes usan los métodos de `EDA` |
| Núcleo | `ai_framework/agents/agent.py` | `Agent(ABC)` (alias `Agente`): ciclo `prepare` → `propose` → `react` → `next_events` |
| Núcleo | `ai_framework/agents/eda_agent.py` | `EDAAgent` (alias `AgenteEDA`): reglas proactivas y `suggest_configuration()` |
| Núcleo | `ai_framework/agents/clustering_agent.py` | `ClusteringAgent` (alias `AgenteClustering`): compara K-Means y HAC-ward y describe los grupos |
| Núcleo | `ai_framework/agents/supervised_agent.py` | `SupervisedAgent`: benchmark con validación cruzada y reglas comunes |
| Núcleo | `ai_framework/agents/regression_agent.py` | `RegressionAgent` (alias `AgenteRegresion`): mejor regresor por RMSE |
| Núcleo | `ai_framework/agents/classification_agent.py` | `ClassificationAgent` (alias `AgenteClasificacion`): mejor clasificador por F1 macro |
| Núcleo | `ai_framework/agents/tools/` | Funciones del framework expuestas al LLM con esquema JSON |
| Núcleo | `ai_framework/agents/events.py` | `EventBus`: `DATASET_LOADED` → `EDA_DONE` → … |
| Núcleo | `ai_framework/agents/llm/` | `OpenAICompatibleProvider` y `FakeLLMProvider` para pruebas |
| Resultados | `ai_framework/utils/results.py` | `Finding`, `ModelingSuggestion` y `ClusteringSuggestion` (inmutables; alias `Hallazgo`, `SugerenciaModelado`, `SugerenciaClustering`) |
| Configuración | `ai_framework/utils/agents/agent_config.py` | Proveedores, agentes, `EDAThresholds` y `ClusteringThresholds`, sin Streamlit |
| Interfaz | `ai_framework/ui/controllers/agents_controller.py` | Casos de uso: iniciar agentes y aplicar sugerencias |
| Interfaz | `ai_framework/ui/models/agents_state.py` | Invalidación de sesiones por dataset y configuración |
| Interfaz | `ai_framework/ui/views/agents_view.py` | Controles y vistas; los textos están en `ui/i18n.py` |
| CLI | `scripts/run_agents.py` (`scripts/ejecutar_agentes.py`) | Mismo flujo sin interfaz |

**Proactivo**: al cargarse el dataset el agente aplica reglas deterministas
(duplicados, nulos, constantes, identificadores, fuga de información y
desbalance) sin gastar cuota del LLM. **Reactivo**: responde preguntas usando
las herramientas. Si el LLM falla, las reglas y los eventos siguen funcionando.

## Agente de clustering

Está en **Clustering → Agente IA** y usa las mismas variables, escalado y máximo
de filas que K-Means y HAC. Sólo agrupa al pulsar **Iniciar agente**: compara
K-Means y HAC-ward de k=2 a k=10 y recomienda la partición con mayor silhouette,
sin gastar cuota del LLM. HAC con otros métodos queda fuera de esa búsqueda
porque suele premiar particiones degeneradas, pero el LLM puede evaluarlos.

Avisa de estructura débil (silhouette menor que 0.25), clusters con menos de 5 %
de las filas, falta de estandarización, identificadores y columnas con muchos
nulos. El LLM recibe tamaños, métricas y las variables más distintivas de cada
grupo en desviaciones estándar; de los grupos con menos de tres filas sólo recibe
el tamaño. Las asignaciones por fila no se envían.

```powershell
.venv\Scripts\python.exe -m scripts.ejecutar_agentes datos.csv --agent clustering
```

## Agentes de regresión y clasificación

Están en **Regresión → Agente IA** y **Clasificación → Agente IA**, y usan el
target, las predictoras y la partición de **Configuración**. Sólo entrenan al
pulsar **Iniciar agente**. Corren el mismo benchmark del framework con validación
cruzada de 5 pliegues dentro de train: regresión evalúa los nueve regresores con
su catálogo de configuraciones; clasificación evalúa una configuración estándar
de cada uno de los siete clasificadores (la búsqueda completa de hiperparámetros
sigue en **Evaluación Comparativa**). La prueba sólo evalúa al ganador.

Reglas proactivas comunes: candidatos que fallaron y pocas filas de entrenamiento
por variable. Regresión: R² de prueba ≤ 0 (no supera a la media), R² débil,
sobreajuste (R² de train muy por encima del de prueba), RMSE inestable entre
pliegues y prueba alejada de la validación. Clasificación: accuracy que no supera
a la clase mayoritaria, sobreajuste en F1 macro, accuracy inflada por desbalance,
clases que se detectan menos de la mitad de las veces y F1 inestable entre
pliegues. Los umbrales están en `SupervisedThresholds`.

Herramientas del LLM: ranking de candidatos, entrenar un algoritmo con la misma
partición, importancia de variables y residuos (regresión) o matriz de confusión
(clasificación). Al LLM sólo llegan métricas y resúmenes, no predicciones por fila.

Encadenamiento: los dos escuchan también `EDA_DONE`. Si el agente EDA detecta la
tarea, el agente que corresponde entrena con las predictoras que el EDA no
desaconsejó. En la CLI se activa con `--agent pipeline`:

```powershell
.venv\Scripts\python.exe -m scripts.ejecutar_agentes datos.csv --target y --agent regression
.venv\Scripts\python.exe -m scripts.ejecutar_agentes datos.csv --target y --agent pipeline
```

En ese modo la exclusión es automática: una predictora legítima con |r| ≥ 0.98
con el target también queda fuera por la regla de fuga. Revise los hallazgos
del EDA antes de confiar en el modelo.

## Proveedores

`gemini`, `groq`, `openrouter` y `ollama` usan el mismo cliente `openai` con
otra `base_url`. `rules` trabaja sin red ni API key.

```powershell
$env:GEMINI_API_KEY = "su-key"
$env:AGENTES_MODELO = "gemini-2.5-flash"
```

En la aplicación también puede usar `.streamlit/secrets.toml`, que está
ignorado por git. Los nombres de modelos gratuitos cambian; verifíquelos en
cada proveedor.

## Uso

En la aplicación: **Exploración de datos → Agente IA**, elija variable objetivo
y LLM, y pulse **Iniciar agente**. **Aplicar configuración sugerida** abre
Configuración con tarea, target y predictoras; no entrena.

```powershell
.venv\Scripts\python.exe -m scripts.ejecutar_agentes data/lab03/concrete/concrete.csv --target strength
.venv\Scripts\python.exe -m scripts.ejecutar_agentes datos.csv --target y --provider gemini --interactive
```

Los flags en español (`--agente`, `--proveedor reglas`, `--interactivo`…) siguen
funcionando como alias.

Cada llamada al LLM se imprime en la consola donde corre Streamlit o el script:
los mensajes enviados, la respuesta o las herramientas pedidas, y los errores.
Los esquemas de las herramientas también se envían, pero no se imprimen.

## Idioma y progreso

Los agentes hablan el idioma de la interfaz (selector Español/English). Los
hallazgos, los avisos de progreso y los textos de respaldo salen de
`ai_framework/agents/messages.py`, que tiene cada mensaje en ambos idiomas. Al
LLM se le pide responder en ese idioma o, si el usuario escribe en otro, en el
del usuario; el idioma se vuelve a leer antes de cada pregunta del chat. Los
prompts de rol están en inglés y la instrucción de idioma se agrega al final.

Mientras trabaja, cada agente avisa al terminar cada paso (`Agent.report`): perfil
listo, K-Means evaluado, comparación de modelos lista, reglas aplicadas, consulta
al LLM y cada herramienta que usa, aviso a otros agentes y fin. En la aplicación
esos avisos aparecen en vivo en un cuadro de estado y quedan en **Actividad del
agente**; en la CLI se imprimen con `  - `. Se elige el idioma con `--language en`
(o `--idioma`).

Para un idioma nuevo basta agregar su bloque en `MESSAGES` y su nombre en
`LANGUAGE_NAMES`; si falta un mensaje se usa el español.

## Agregar un agente

1. Herramientas en `ai_framework/agents/tools/<area>_tools.py` con `@tool`.
2. Subclase de `Agent` en `ai_framework/agents/<area>_agent.py`: `listens_to`,
   `tools`, `propose()` (obligatorio) y `next_events()`. Un agente de
   regresión, por ejemplo, escucharía `EDA_DONE` y usaría `ejecutar_experimento`.
3. Su `AgentConfig` en `ai_framework/utils/agents/agent_config.py`.
4. Caso de uso en `agents_controller.py`, vista en `agents_view.py`, ruta en
   `view_catalog.py`, botón en `navigation.py`, destino en `app.py` y textos en
   español e inglés en `i18n.py`.
5. Pruebas con `FakeLLMProvider`, sin red.
