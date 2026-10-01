# Lab01 — Métodos no supervisados

Implementación y análisis de ACP, HAC, K-Means, t-SNE y UMAP con el framework
incluido. El ejecutor utiliza el paquete local; no requiere el repositorio de
origen. La interfaz Streamlit es opcional y no es necesaria para reproducir
los experimentos.

## Contenido

| Archivo o carpeta | Descripción |
|---|---|
| `app.py` | Punto de entrada de la interfaz Streamlit |
| `data/ejemplo_analisis.csv` | Conjunto de 16 observaciones, cuatro variables numéricas y la categoría `segmento` |
| `framework_ia/` | Framework local: 31 archivos Python, incluida la clase `Cluster` |
| `scripts/ejecutar_lab01.py` | Ejecutor de los cinco métodos y exportación de sus resultados |
| `resultados/` | 41 CSV y dos JSON: configuración, comparaciones, proyecciones, asignaciones y resumen |
| `informe/analisis_lab1.pdf` | Informe en PDF |
| `informe/plantilla_informe.tex` | Código fuente del informe, con tablas y gráficos incorporados |
| `informe/referencias.bib` | Bibliografía de respaldo para reutilización |

## Instalación

Requiere Python 3.12 y acceso a las dependencias de `requirements.txt`.
Abra PowerShell en la raíz de esta carpeta y ejecute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Los comandos utilizan directamente el intérprete del entorno; no es necesario
activar scripts de PowerShell. En otro equipo se crea un entorno nuevo.

## Reproducción de los experimentos

```powershell
.\.venv\Scripts\python.exe -B -m scripts.ejecutar_lab01 data/ejemplo_analisis.csv --salida resultados_reproducidos --semilla 42
```

La salida se guarda por separado para conservar los resultados incluidos en
`resultados/`. Una ejecución sobre una carpeta existente actualiza archivos con
los mismos nombres. Los argumentos disponibles se consultan con:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.ejecutar_lab01 --help
```

El procesamiento predeterminado utiliza las cuatro variables numéricas,
imputación por mediana y estandarización; `segmento` no interviene en el ajuste.
Las configuraciones incompatibles con 16 observaciones se registran como
omisiones, por ejemplo, t-SNE con perplexity 30.

El ejecutor termina con código distinto de cero si ocurre un error. El resumen
contiene el estado global, las configuraciones evaluadas de los cinco métodos,
las selecciones y el inventario de archivos:

```powershell
Get-Content .\resultados_reproducidos\resumen_experimentos.json
```

El estado global esperado es `completado`. Las versiones de las bibliotecas y
la huella de los datos quedan registradas en los JSON; diferencias de versiones
pueden influir en la reproducción numérica.

## Interfaz Streamlit

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --global.developmentMode false
```

Abra `http://127.0.0.1:8501/` en un navegador y mantenga la terminal abierta.
La interfaz permite seleccionar el CSV y explorar los métodos del framework.
Para detener el servidor utilice `Ctrl + C`; para iniciarlo de nuevo repita el
comando desde la raíz de la carpeta.

## Informe

El PDF se encuentra en `informe/analisis_lab1.pdf`. El archivo
`informe/plantilla_informe.tex` puede compilarse de forma independiente con
pdfLaTeX, por ejemplo, en Overleaf. Incluye gráficos y referencias en su código;
`referencias.bib` se conserva como respaldo y no es un requisito de compilación.

Ejecutar los experimentos no modifica ni recompila el informe. Si se cambian los
datos o los parámetros, las tablas, visualizaciones y conclusiones del documento
requieren una revisión independiente.

## Autores

- Dylan Elizondo Alvarado — 504610652
- Joseph Garcia Montero — 402680547
- Rodrigo Ureña Castillo — 118910482
- Mariana Ureña Padilla — 119180324