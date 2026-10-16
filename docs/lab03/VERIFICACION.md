# Verificación reproducible de Lab03

El 16 de octubre de 2026 se ejecutó dos veces el benchmark completo sobre la
referencia local de scikit-learn. Ambas CLI terminaron con código **0**:
**14 candidatos correctos, nueve familias evaluadas y ningún fallo de prueba**.
Estos resultados son evidencia técnica de esta ejecución, no un informe
académico ni una conclusión clínica.

## Entorno observado

| Componente | Versión |
|---|---|
| Python | 3.12.14 |
| numpy / pandas | 2.5.4 / 3.0.6 |
| scipy / scikit-learn | 1.18.1 / 1.9.1 |
| Streamlit / Plotly | 1.66.0 / 7.1.0 |
| pytest / Ruff | 9.1.1 / 0.17.1 |

`python -m pip check` no detectó dependencias incompatibles. Se limitaron
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y `NUMBA_NUM_THREADS` a `2`.
`requirements-lab02-lock.txt` conserva la evidencia histórica de Lab02; no es
una captura de este entorno. Versiones futuras o plataformas distintas pueden
cambiar resultados numéricos: conserve las versiones exportadas al entrenar.

## Reproducción local

Desde la raíz, en PowerShell con el entorno `.venv` instalado:

```powershell
$env:OMP_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
$env:NUMBA_NUM_THREADS="2"
$raiz = Join-Path $env:TEMP "lab03-reproduccion"
.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab03 --salida "$raiz/data"
.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab03 --salida "$raiz/data"
.venv\Scripts\python.exe -m scripts.ejecutar_lab03 "$raiz/data/diabetes_regresion.csv" --target target --feature-simple bmi --salida "$raiz/run1" --fuente https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
.venv\Scripts\python.exe -m scripts.ejecutar_lab03 "$raiz/data/diabetes_regresion.csv" --target target --feature-simple bmi --salida "$raiz/run2" --fuente https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
```

Use un directorio nuevo si ya contiene resultados. La preparación repetida no
reescribe archivos idénticos; la CLI rechaza salidas no vacías. No hay descargas,
transmisiones ni entrenamiento durante la aplicación de la fuente en la UI.

En la ejecución observada, los artefactos quedaron en el directorio temporal
`lab03-reference-20261016-130250/{data,run1,run2}` y el resumen local en
`verificacion.json`. No se incorporaron resultados generados al repositorio.

## Datos y protocolo

- 442 filas; diez predictoras sin escalar y objetivo `target`; RLS usa `bmi`.
- CSV materializado: **25 677 bytes**, SHA-256
  `d0b14a7a6a4015e4291e82705a7dd34906afb0b87bf5f67037bf1ec2f51e663f`.
  Es el hash del CSV UTF-8/LF/sin índice, no del paquete original de sklearn.
- Semilla 42, 331 filas de train y 111 de prueba. Cinco pliegues externos
  mezclados sobre train, con validaciones de 67/66/66/66/66 filas.
- Selección por media RMSE de validación; desviación muestral (`ddof=1`).
  Imputación/escala se ajustan por train de cada pliegue externo.
- LassoCV/RidgeCV mantienen la limitación de preprocesamiento interno documentada
  en [README](README.md#validación-interna-y-reproducibilidad). Sus scores
  internos no son evidencia externa independiente.

## Resultados observados

Una configuración por familia, seleccionada sólo por validación. RMSE y MAE se
expresan en unidades del objetivo; R² no es un porcentaje. MSE y los valores sin
redondear están en los JSON exportados.

| Familia | Configuración | RMSE validación ± DE | RMSE prueba | MAE prueba | R² prueba |
|---|---|---:|---:|---:|---:|
| RLS | Estándar, bmi | 63.298420 ± 4.183195 | 61.446051 | 49.805052 | 0.317210 |
| RLM | Estándar, diez predictoras | 56.118567 ± 3.502394 | 53.369567 | 41.548507 | 0.484906 |
| Lasso | alpha=0.1 | 56.055613 ± 3.573092 | 53.281974 | 41.484718 | 0.486595 |
| LassoCV | Cuadrícula nativa | 56.173458 ± 3.852980 | 53.281974 | 41.484718 | 0.486595 |
| Ridge | alpha=1 | 56.071072 ± 3.575982 | 53.318239 | 41.507342 | 0.485896 |
| RidgeCV | Cuadrícula nativa | 56.157320 ± 3.706929 | 53.318239 | 41.507342 | 0.485896 |
| SVR | C=10 | 59.579515 ± 3.453714 | 51.915118 | 41.334239 | 0.512598 |
| DT | Profundidad 5 | 70.610743 ± 6.814127 | 62.981947 | 48.493480 | 0.282650 |
| RF | 100 árboles, profundidad 5 | 60.716052 ± 3.381659 | 54.024524 | 43.092129 | 0.472186 |

**Lasso alpha=0.1** fue el ganador por validación. Aunque SVR tuvo menor RMSE
de prueba en esta partición, eso **no cambia la selección**. Una única semilla,
un dataset y diferencias pequeñas de validación no prueban superioridad general.

## Igualdad entre repeticiones

Se compararon los **39 archivos** de cada ejecución: 29 fueron idénticos byte
a byte, incluyendo comparación, métricas y todas las predicciones. Los diez
`configuracion.json` difieren únicamente en `argumentos.salida`, porque sus
directorios son distintos. Al excluir sólo ese campo, todos los JSON coinciden
completamente: índices, pliegues, versiones, parámetros, selección y metadatos.
No se eliminaron scores, semillas ni campos de procedencia para comparar.

## Alcance de la verificación

Comprobaciones locales finales con los límites de hilos indicados:

- Suite completa: `.venv/Scripts/python.exe -m pytest -q` → **217 passed,
  152 subtests passed**, 156.69 s, código 0.
- Casos de referencia, UI/CLI Lab03 y UI/preparación Lab02: **98 passed**,
  50.41 s. Los diez casos nuevos verifican datos crudos/offline, determinismo,
  conflictos antes de escribir, idempotencia, códigos CLI, aplicación explícita,
  preset de un solo uso, recuperación y procedencia congelada.
- Ruff/format de los tres módulos Python nuevos, nombres no definidos (`F821`)
  del entrypoint, ayudas CLI y `git diff --check`: correctos.
- Ruff completo del entrypoint conserva 15 avisos anteriores: se compararon
  códigos/mensajes contra `HEAD` y coinciden, sin avisos nuevos. No se corrigió
  estilo ajeno a este trabajo ni se declaró ese archivo totalmente limpio.
- Auditoría estática premium estricta intentada sin escrituras; permaneció
  sin resultado al recorrer el entorno virtual y se interrumpió. No se declara
  aprobada ni sustituye los checks funcionales o de navegador.

La CI conserva Lab02, añade `lab_03`, comprueba las ayudas CLI y prepara/ejecuta
esta referencia offline. La definición se verificó localmente; no se ejecutó
GitHub Actions ni se publicó una rama. Las pruebas funcionales y de navegador
se registran al cerrar el trabajo; AppTest por sí solo no certifica apariencia,
teclado o interacción real en el navegador.
