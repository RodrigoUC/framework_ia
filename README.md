# Framework de análisis exploratorio y aprendizaje automático

Framework modular para cargar cualquier archivo CSV, preparar sus datos y
ejecutar EDA, ACP, K-Means, K-Medoids, clustering jerárquico, t-SNE, UMAP y clasificación.
La interfaz gráfica usa Streamlit y Plotly para los gráficos interactivos,
mientras que los algoritmos permanecen en clases independientes y
reutilizables. La versión oficial del proyecto es este directorio
(`framework_start`).

## Inicio rápido

```bash
cd framework_start
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Anaconda también puede usarse `conda activate base` (o el entorno donde
estén instaladas las dependencias) antes de ejecutar los comandos.

En la barra lateral:

1. Seleccione un CSV ubicado en `framework_start` o suba uno desde el navegador.
2. Ajuste separador, decimal, codificación e índice según el archivo.
3. Navegue por Datos, Clustering, Clasificación, Resultados o Regresión.
4. Seleccione una técnica; su configuración aparece en la vista correspondiente.
5. Ejecute el análisis desde su vista. En Clasificación, abra **Resultados de
   clasificación** o **Comparar modelos entrenados** para revisar sólo
   ejecuciones individuales guardadas, sin volver a entrenar. Random Forest
   conserva resultados independientes para `gini` y `entropy`; la comparación
   muestra accuracy general y recall porcentual por clase real.

En la vista **Datos y preparación** también puede:

- Ver el tipo analítico de cada columna (numérica/categórica), sus nulos y
  su cardinalidad.
- Eliminar columnas manualmente antes de aplicar el resto de la preparación.
- Calcular una partición train/test/validación eligiendo el porcentaje de
  cada subconjunto y, de forma opcional, una columna para estratificar. Esa
  partición queda disponible para Clasificación (y, a futuro, Regresión),
  que pueden reutilizarla en vez de dividir los datos de nuevo.

EDA conserva sus gráficos e incorpora ACP como análisis explícito. K-Means y
K-Medoids tienen acciones independientes; t-SNE y UMAP son proyecciones
opcionales dentro de K-Means, no algoritmos de agrupamiento. La regresión
aparece como pilar independiente, marcada como próxima mientras sus métodos
sigan siendo una plantilla.

El archivo `data/ejemplo_analisis.csv` permite comprobar la aplicación de inmediato;
puede reemplazarse por cualquier otro CSV.

No existen nombres de columnas ni rutas de datasets codificados en los modelos.
Por ello, reemplazar el CSV no exige modificar el código.

## Arquitectura

```text
app.py                         Punto de entrada de Streamlit.
framework_ia/
  datos/                       Carga, EDA, preprocesamiento y partición.
  modelos/
    supervisado/               Clasificación y regresión.
    no_supervisado/            Agrupamiento y reducción dimensional.
  resultados.py                Objetos de transferencia entre capas.
  visualizacion.py             Figuras Plotly independientes de Streamlit.
  ui/streamlit_app.py          Interfaz y estado de sesión.
scripts/                       Automatización de Lab01 y empaquetado.
tests/                         Pruebas automatizadas.
data/                          Datasets locales de ejemplo.
docs/lab01/                    Guías, plantilla y referencias del laboratorio.
docs/lab02/                    Protocolo y reproducción de clasificación.
```

La separación aplica responsabilidad única y composición: Streamlit no calcula
modelos, los modelos no leen widgets y las visualizaciones reciben resultados
ya calculados.

## Dependencias

Todos los gráficos del framework (EDA, ACP, clustering, t-SNE/UMAP y
clasificación) se generan con Plotly, incluido en `requirements.txt`. UMAP
está incluido mediante `umap-learn`. K-Medoids se implementa en
`framework_ia/modelos/algoritmos_cluster.py`, por lo que no requiere una
biblioteca adicional.

Las versiones mínimas están en `requirements.txt`. LAB 02 requiere Streamlit
1.65 o posterior para conservar controles al navegar y describir las tablas
y gráficos de forma accesible. El código de t-SNE admite
las versiones de scikit-learn que nombran el parámetro de iteraciones como
`max_iter` y las versiones anteriores que lo nombran `n_iter`.

## Estado de las clases

| Módulo | Clase o capacidad | Estado |
| --- | --- | --- |
| `framework_ia/datos` | Operaciones tabulares, EDA y preprocesamiento | Implementada |
| `framework_ia/modelos/no_supervisado/agrupamiento.py` | K-Means, K-Medoids, HAC y benchmarks | Implementada |
| `framework_ia/modelos/no_supervisado/agrupamiento.py` | ACP, t-SNE y UMAP | Implementada; UMAP depende de `umap-learn` |
| `framework_ia/visualizacion.py` | Figuras Plotly | Implementada |
| `framework_ia/modelos/supervisado/clasificacion.py` | KNN, DT, RF, XGBoost, AdaBoost y NR | Implementada |
| `framework_ia/modelos/supervisado/regresion.py` | `Regresion.RLS()`, `RLM()` y `RL()` | Plantilla declarada |

`Clasificacion` incluye los cinco algoritmos del LAB 02, comparación por
validación y evaluación en test reservado; también conserva Naive Bayes para
entrenamiento individual. Regresión permanece pendiente.

## Uso desde Python

```python
from framework_ia.datos import EDA
from framework_ia.modelos import Cluster

eda = EDA(ruta_datos="mis_datos.csv")
datos = eda.preparar_dataset()

cluster = Cluster(dataframe=datos, features=["variable_1", "variable_2"])
resultado_kmeans = cluster.K_means(n_clusters=3)
# También están disponibles los nombres pythonicos:
resultado_kmeans = cluster.k_means(n_clusters=3)

reduccion = Cluster(
    dataframe=datos,
    features=["variable_1", "variable_2"],
)
resultado_acp = reduccion.ACP(n_componentes=2)
# Equivalente pythonico:
resultado_acp = reduccion.acp(n_componentes=2)
```

## Validación

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Las pruebas cubren carga de CSV, limpieza, preprocesamiento, ACP, t-SNE,
UMAP, clustering, herencia, existencia de métodos pendientes y alias pythonicos.

## Experimentos reproducibles de Lab01

Para ejecutar las configuraciones base y sus variaciones sobre el CSV del
profesor, use el siguiente comando:

```bash
python -m scripts.ejecutar_lab01 data/datos_profesor.csv --salida salida_lab01 --semilla 42
```

El directorio de salida contiene la configuración exacta (`configuracion.json`),
un resumen (`resumen_experimentos.json`), las comparaciones de ACP, K-Means,
HAC, t-SNE y UMAP, y las asignaciones de los modelos seleccionados. La métrica
``trustworthiness`` para t-SNE/UMAP sirve de apoyo: la elección final debe
justificarse también con las visualizaciones y el contexto del dataset.

UMAP inicializa automáticamente un directorio de caché temporal para Numba,
por lo que puede ejecutarse incluso cuando el caché por defecto no es
escribible. Se recomienda usar Python 3.11 o 3.12 para un entorno estable con
las dependencias científicas del laboratorio.

## Referencia de implementación

La organización funcional toma como caso de estudio el paquete y el notebook
entregados para la clase, pero reemplaza su diseño monolítico por componentes
independientes.

Murillo-Morera, J. D. (2026). *Paquete 1: Análisis de datos exploratorios (EDA)*
[Código fuente de curso no publicado]. Universidad Nacional de Costa Rica.

## LAB 02: clasificación

Consulte [la guía de implementación y reproducción](docs/lab02/README.md).
Incluye KNN, árboles, Random Forest, XGBoost y AdaBoost, configuración estándar
y variantes, selección por validación y test reservado, interfaz separada y CLI.
La interfaz también ofrece Naive Bayes como modelo individual compatible;
no forma parte de las cinco familias del experimento del laboratorio.
Los CSV de Kaggle permanecen locales; no son automáticamente los archivos
exactos del profesor. No se incluyen resultados académicos inventados.
