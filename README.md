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
| `framework_ia/ui/modelos/` | Capa de modelos de estado, catálogo de vistas, disponibilidad y validación de procedencia |
| `framework_ia/ui/controladores/` | Capa de casos de uso y controladores de fuentes, no supervisado, clasificación y regresión |
| `framework_ia/ui/vistas/` | Capa de presentación en Streamlit (tema, navegación, controles y vistas por dominio) |
| `framework_ia/ui/streamlit_app.py` | Punto de entrada y enrutador principal de la interfaz **Nexo IA** |
| `scripts/` | Ejecución reproducible por línea de comandos, informes y empaquetado |
| `tests/` | Suite de pruebas unitarias e integrales del framework y la interfaz |

## Ejecutar la aplicación y pruebas

En PowerShell, dentro de la carpeta del proyecto (Python 3.11 o superior; verificado con Python 3.13):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Ejecución por línea de comandos (`scripts/`)

### 1. Agrupamiento (No Supervisado)

```powershell
python -m scripts.ejecutar_no_supervisado data/ejemplo_analisis.csv --salida salida_no_supervisado --semilla 42
```

### 2. Clasificación (Potabilidad y Diabetes)

Integrantes: Dylan Elizondo Alvarado (504610652), Joseph Garcia Montero (402680547), Rodrigo Ureña Castillo (118910482) y Mariana Ureña Padilla (119180324).

```powershell
python -m scripts.preparar_potabilidad data/lab02/water_potability.csv
python -m scripts.preparar_diabetes data/lab02/diabetes_original.csv
python -m scripts.ejecutar_clasificacion data/lab02/potabilidad.csv --target Potability --salida salida_reproducida/potabilidad --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
python -m scripts.ejecutar_clasificacion data/lab02/diabetes.csv --target diabetes --features chol stab.glu hdl ratio location age gender height weight frame bp.1s bp.1d bp.2s bp.2d waist hip time.ppn --incluir-categoricas --salida salida_reproducida/diabetes --semilla 42 --fuente https://hbiostat.org/data/repo/diabetes
python -m scripts.generar_informe salida_reproducida
```

### 3. Regresión (Diabetes y Concrete)

```powershell
python -m scripts.preparar_diabetes
python -m scripts.ejecutar_regresion data/lab03/diabetes_regresion.csv --target target --feature-simple bmi --salida salida_lab03
python -m scripts.preparar_concrete Concrete_Data.xls
python -m scripts.ejecutar_regresion data/lab03/concrete/concrete.csv --target strength --feature-simple cement --salida salida_lab03/concrete --fuente https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
python -m scripts.generar_informe salida_lab03/concrete --datos data/lab03/concrete/concrete.csv --procedencia data/lab03/concrete/procedencia.json --salida informe/main.tex
```

### 4. Empaquetado de entrega

```powershell
python -m scripts.preparar_entrega --ids ID-1 ID-2 --resultados salida_no_supervisado --informe informe.pdf
```

- [Preparación y variables de Concrete](data/lab03/concrete/README.md).
- [Experimentos, informe y resultados verificados](docs/lab03/CONCRETE.md).
- [API y flujos de regresión](docs/lab03/README.md).
