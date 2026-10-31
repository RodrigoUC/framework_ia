# Agentes

Los agentes coordinan el framework con un modelo de lenguaje (LLM) gratuito.
El LLM decide qué analizar e interpreta; los cálculos siempre los hace el
framework. Nunca se genera ni ejecuta código, y al LLM sólo llega un resumen
del dataset, no sus filas.

## Arquitectura

Sigue las mismas capas que el resto del framework:

| Capa | Módulo | Rol |
| --- | --- | --- |
| Núcleo | `agentes/pizarra.py` | `Pizarra(EDA)`: estado compartido; los agentes usan los métodos de `EDA` |
| Núcleo | `agentes/base.py` | `Agente(ABC)`: ciclo preparar → proponer → reaccionar → siguientes |
| Núcleo | `agentes/exploracion.py` | `AgenteEDA`: reglas proactivas y `sugerir_configuracion()` |
| Núcleo | `agentes/agrupamiento.py` | `AgenteClustering`: compara K-Means y HAC-ward y describe los grupos |
| Núcleo | `agentes/herramientas/` | Funciones del framework expuestas al LLM con esquema JSON |
| Núcleo | `agentes/eventos.py` | `BusEventos`: `DATASET_CARGADO` → `EDA_LISTO` → … |
| Núcleo | `agentes/llm/` | Proveedor compatible con OpenAI y `ProveedorFalso` para pruebas |
| Resultados | `resultados.py` | `Hallazgo`, `SugerenciaModelado` y `SugerenciaClustering` (inmutables) |
| Configuración | `utils/configuracion_agentes.py` | Proveedores, agentes, `UmbralesEDA` y `UmbralesClustering`, sin Streamlit |
| Interfaz | `ui/configuracion_agentes.py`, `ui/estado_agentes.py`, `ui/agentes.py` | Controles, invalidación por dataset y vista |
| CLI | `scripts/ejecutar_agentes.py` | Mismo flujo sin interfaz |

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
.venv\Scripts\python.exe -m scripts.ejecutar_agentes datos.csv --agente clustering
```

## Proveedores

`gemini`, `groq`, `openrouter` y `ollama` usan el mismo cliente `openai` con
otra `base_url`. `reglas` trabaja sin red ni API key.

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
.venv\Scripts\python.exe -m scripts.ejecutar_agentes datos.csv --target y --proveedor gemini --interactivo
```

Cada llamada al LLM se imprime en la consola donde corre Streamlit o el script:
los mensajes enviados, la respuesta o las herramientas pedidas, y los errores.
Los esquemas de las herramientas también se envían, pero no se imprimen.

## Agregar un agente

1. Herramientas en `agentes/herramientas/<area>.py` con `@herramienta`.
2. Subclase de `Agente` en `agentes/<area>.py`: `escucha`, `herramientas`,
   `proponer()` (obligatorio) y `siguientes()`. Un agente de regresión, por
   ejemplo, escucharía `EDA_LISTO` y usaría `ejecutar_experimento`.
3. Su `ConfiguracionAgente` en `utils/configuracion_agentes.py` y una vista con
   botón «Agente IA» en la sección del menú que le corresponda.
4. Pruebas con `ProveedorFalso`, sin red.
