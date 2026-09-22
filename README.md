# Framework de análisis exploratorio y clustering

Framework modular para cargar cualquier archivo CSV, preparar sus datos y
ejecutar EDA, ACP, K-Means, K-Medoids, clustering jerárquico, t-SNE y UMAP.
La interfaz gráfica usa Streamlit, mientras que los algoritmos permanecen en
clases independientes y reutilizables. La versión oficial del proyecto es
este directorio (`framework_start`); las carpetas `Framework` y los archivos
de la raíz son versiones anteriores o material de referencia.

## Inicio rápido

```bash
cd framework_start
python -m pip install -r requirements.txt
python -m streamlit run vca.py
```

En Anaconda también puede usarse `conda activate base` (o el entorno donde
estén instaladas las dependencias) antes de ejecutar los comandos.

En la barra lateral:

1. Seleccione un CSV ubicado en `framework_start` o suba uno desde el navegador.
2. Ajuste separador, decimal, codificación e índice según el archivo.
3. Elija las variables de análisis.
4. Active la codificación categórica si desea incluir columnas de texto.
5. Ejecute el análisis requerido desde sus pestañas.

El archivo `ejemplo_analisis.csv` permite comprobar la aplicación de inmediato;
puede reemplazarse por cualquier otro CSV.

No existen nombres de columnas ni rutas de datasets codificados en los modelos.
Por ello, reemplazar el CSV no exige modificar el código.

## Arquitectura

```text
fuentes_datos.py          Carga configurable desde ruta o memoria.
DataFrame.py              Operaciones tabulares, limpieza y EDA.
eda.py                    Pipeline general de preparación.
preprocesamiento.py       Imputación, codificación y escala para modelos.
no_supervisado.py         Contrato y comportamiento compartido.
reduccion_dimensional.py  ACP, t-SNE y UMAP.
algoritmos_cluster.py     Implementación independiente de K-Medoids.
cluster.py                K-Means, K-Medoids y HAC.
resultados.py             Objetos de transferencia entre capas.
visualizacion.py          Figuras Matplotlib sin dependencia de Streamlit.
vca.py                    Interfaz y estado de sesión de Streamlit.
```

La separación aplica responsabilidad única y composición: Streamlit no calcula
modelos, los modelos no leen widgets y las visualizaciones reciben resultados
ya calculados.

## Dependencias

UMAP está incluido en `requirements.txt` mediante `umap-learn`. K-Medoids se
implementa en `algoritmos_cluster.py`, por lo que no requiere una biblioteca
adicional.

Las versiones mínimas están en `requirements.txt`. El código de t-SNE admite
las versiones de scikit-learn que nombran el parámetro de iteraciones como
`max_iter` y las versiones anteriores que lo nombran `n_iter`.

## Estado de las clases

| Módulo | Clase o capacidad | Estado |
| --- | --- | --- |
| `DataFrame.py` | Operaciones tabulares y EDA | Implementada |
| `eda.py` | Preparación y calidad del dataset | Implementada |
| `preprocesamiento.py` | Imputación, codificación y escala | Implementada |
| `cluster.py` | K-Means, K-Medoids, HAC y benchmarks | Implementada |
| `reduccion_dimensional.py` | ACP, t-SNE y UMAP | Implementada; UMAP depende de `umap-learn` |
| `visualizacion.py` | Figuras Matplotlib/Seaborn | Implementada |
| `clasificacion.py` | `Clasificacion.RF()` y `Clasificacion.NR()` | Implementada |
| `progresion.py` | `Progresion.RLS()`, `RLM()` y `RL()` | Plantilla declarada |

Las clases supervisadas y sus métodos existen para completar el framework por
etapas; actualmente `Clasificacion` (RF/NR) y su flujo de métricas y predicciones
ya están funcionales.

## Uso desde Python

```python
from eda import EDA
from cluster import Cluster
from reduccion_dimensional import ReduccionDimensional

eda = EDA(ruta_datos="mis_datos.csv")
datos = eda.preparar_dataset()

cluster = Cluster(dataframe=datos, features=["variable_1", "variable_2"])
resultado_kmeans = cluster.K_means(n_clusters=3)
# También están disponibles los nombres pythonicos:
resultado_kmeans = cluster.k_means(n_clusters=3)

reduccion = ReduccionDimensional(
    dataframe=datos,
    features=["variable_1", "variable_2"],
)
resultado_acp = reduccion.ACP(n_componentes=2)
# Equivalente pythonico:
resultado_acp = reduccion.acp(n_componentes=2)
```

## Validación

```bash
python -m unittest discover -s tests -v
```

Las pruebas cubren carga de CSV, limpieza, preprocesamiento, ACP, t-SNE,
clustering, herencia, existencia de métodos pendientes y alias pythonicos.

## Referencia de implementación

La organización funcional toma como caso de estudio el paquete y el notebook
entregados para la clase, pero reemplaza su diseño monolítico por componentes
independientes.

Murillo-Morera, J. D. (2026). *Paquete 1: Análisis de datos exploratorios (EDA)*
[Código fuente de curso no publicado]. Universidad Nacional de Costa Rica.
