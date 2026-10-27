# Nexo IA (`framework_ia`) · Plataforma Unificada de Análisis de Datos y Machine Learning

`framework_ia` es un producto unificado de análisis exploratorio de datos, aprendizaje no supervisado y aprendizaje supervisado (clasificación y regresión), desacoplado de la interfaz gráfica (`ui/streamlit_app.py`).

## Arquitectura del Framework

```text
DataFrame         (framework_ia/datos/dataframe.py · operaciones tabulares y estadísticas)
└── EDA           (framework_ia/datos/eda.py · preparación, calidad y gráficos exploratorios)
    ├── NoSupervisado (framework_ia/modelos/no_supervisado/)
    │      └── Cluster -> ACP, K-Means (+ t-SNE, UMAP), K-Medoids, HAC
    └── Supervisado   (framework_ia/modelos/supervisado/)
           ├── Clasificacion  ->  KNN, DT, RF, XGBoost, AdaBoost, NR, RL
           └── Regresion      ->  RLS, RLM, RL, Lasso, LassoCV, Ridge, RidgeCV, SVR, DT, RF
```

## Módulos Principales

| Ruta | Responsabilidad |
| --- | --- |
| `framework_ia/datos/` | `DataFrame`, `EDA`, `CargadorCSV`, `Particionador`, `PreprocesadorNoSupervisado` y `PreprocesadorSupervisado` |
| `framework_ia/modelos/no_supervisado/` | `NoSupervisado` y `Cluster` (ACP, K-Means, K-Medoids, HAC, t-SNE, UMAP y `benchmark`) |
| `framework_ia/modelos/supervisado/` | `Supervisado`, `Clasificacion` y `Regresion` (`Progresion`) con preprocesamiento ajustado solo en train |
| `framework_ia/visualizacion.py` | `VisualizadorDatos`, `VisualizadorNoSupervisado`, `VisualizadorSupervisado` y `VisualizadorRegresion` |
| `framework_ia/exportacion.py` y `framework_ia/evidencia.py` | Exportación y verificación de evidencia reproducible de experimentos supervisados |
| `framework_ia/ui/` | Interfaz interactiva **Nexo IA** en Streamlit (`streamlit_app.py`) |
| `scripts/` | Ejecución reproducible por línea de comandos, informes y empaquetado |
| `tests/` | Suite de pruebas unitarias e integrales del framework y la interfaz |

## Abrir la aplicación

En PowerShell, dentro de la carpeta del proyecto (Python 3.11 o superior; verificado con Python 3.13):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Caso Regresión · Concrete (Lab03)

El caso Concrete utiliza el cargador CSV existente, los regresores y el benchmark del framework. La preparación conserva el XLS original, retira únicamente duplicados completos y registra procedencia verificable.

- [Preparación y variables de Concrete](data/lab03/concrete/README.md).
- [Experimentos, informe y resultados verificados](docs/lab03/CONCRETE.md).
- [API y flujos de regresión](docs/lab03/README.md).

---

## Caso Clasificación · Potabilidad y Diabetes (Lab02)

Integrantes: Dylan Elizondo Alvarado (504610652), Joseph Garcia Montero (402680547), Rodrigo Ureña Castillo (118910482) y Mariana Ureña Padilla (119180324).

### Protocolo experimental de clasificación

1. Prueba reservada: 25 % estratificado, semilla 42. No participa en ninguna decisión.
2. Validación cruzada estratificada de 5 pliegues sobre el 75 % restante.
3. Cada familia (KNN, DT, RF, XGBoost, AdaBoost) tiene su configuración estándar y tres variantes que cambian un solo hiperparámetro (20 configuraciones).
4. Gana en cada familia la de mayor F1 macro promedio. Esa ganadora se vuelve a evaluar conservando la mitad de los atributos con mayor información mutua y se queda la mejor de las dos.
5. La elegida se reajusta con todo el entrenamiento y se evalúa una vez en prueba.

### Reproducir experimentos y pruebas desde cero

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_potabilidad_lab02 data/lab02/water_potability.csv
.\.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab02 data/lab02/diabetes_original.csv
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 data/lab02/potabilidad.csv --target Potability --salida salida_reproducida/potabilidad --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 data/lab02/diabetes.csv --target diabetes --features chol stab.glu hdl ratio location age gender height weight frame bp.1s bp.1d bp.2s bp.2d waist hip time.ppn --incluir-categoricas --salida salida_reproducida/diabetes --semilla 42 --fuente https://hbiostat.org/data/repo/diabetes
.\.venv\Scripts\python.exe -m pytest -q
```
