# Lab03: regresión y benchmarking

El núcleo de regresión conserva la jerarquía `Supervisado → Regresion` y separa
los modelos de la interfaz. `Progresion` sigue siendo un alias compatible.

## Referencia local sin descargas

En **Fuente de datos → Diabetes · referencia Lab03 → Aplicar fuente** se carga
`sklearn.datasets.load_diabetes(as_frame=True, scaled=False)`. Son 442 filas,
diez predictoras crudas y `target`: una medida cuantitativa de progresión de la
enfermedad al año. No es un diagnóstico binario ni el dataset HBiostat de Lab02.
La nota de estandarización del texto `DESCR` de sklearn describe su opción
predeterminada `scaled=True`; aquí se solicita explícitamente `False`.

Aplicar la fuente configura una sola vez `target`, las diez predictoras,
`feature_simple="bmi"`, 25 % de prueba y semilla 42. No entrena ni imputa/escala
globalmente. Los ajustes manuales posteriores se conservan al navegar. Cambiar
el radio de origen sin pulsar **Aplicar fuente** no cambia los datos activos.
Los CSV arbitrarios siguen disponibles; el núcleo no reconoce nombres de datasets.

```powershell
.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab03
.venv\Scripts\python.exe -m scripts.ejecutar_lab03 data/lab03/diabetes_regresion.csv --target target --feature-simple bmi --salida salida_lab03 --fuente https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
```

La preparación es offline e idempotente: comprueba CSV y procedencia antes de
escribir, conserva archivos ajenos y rechaza contenido distinto. Registra
opciones de carga, esquema, preset y versiones en `data/lab03/procedencia.json`.
Su SHA-256 corresponde al **CSV materializado** (UTF-8, LF, sin índice), no a
los archivos comprimidos originales incluidos en sklearn. La interfaz conserva
esa distinción en sus snapshots/exportaciones. La CLI registra los bytes del
CSV leído y sus opciones; conserve el sidecar de preparación como evidencia de
la materialización. [Detalles de los datos](../../data/lab03/README.md) y
[versiones/resultados realmente observados](VERIFICACION.md).

## Primer entrenamiento

```python
import pandas as pd
from framework_ia import Regresion

datos = pd.read_csv("datos.csv")
regresion = Regresion(dataframe=datos, target="respuesta", features=["x1", "x2"])
resultado = regresion.lineal_multiple()  # Alias: RLM()
print(resultado.metricas)  # rmse, mae, mse, r2 sobre prueba
predicciones = regresion.predecir(datos[["x1", "x2"]])
```

Para una sola predictora numérica, usar `lineal_simple()` o `RLS()` con una
única columna en `features`. `entrenar(algoritmo="RLM", ...)` expone la misma
operación. Si no se indica `target`, se utiliza la última columna; si no se
indican `features`, se seleccionan las demás. `RL()` continúa pendiente y
corresponde a clasificación logística, no a este laboratorio.

## Contratos

| Aspecto | Comportamiento |
|---|---|
| Objetivo | Numérico real, sin nulos, infinitos ni valor constante. También se exige variación en entrenamiento. |
| Partición | 25 % para prueba y semilla 42, sin estratificación. Train y prueba requieren al menos dos filas. |
| Partición externa | `particion=ResultadoParticion(...)` conserva los índices; valida cobertura, solapamientos y cambios del dataset. Validación queda reservada. |
| Preprocesamiento | Imputación por mediana/moda, eliminación de columnas constantes y escala ajustadas sólo con train. Categóricas requieren `incluir_categoricas=True`. |
| Selección de atributos | Desactivada: no se aplica información mutua de clasificación a un objetivo continuo. |
| Inferencia | `predecir(X)` alinea columnas por nombre; exige las predictoras originales y conserva el índice. No vuelve a ajustar. |
| Evaluación | `evaluar()` devuelve el resultado guardado; no entrena. Un entrenamiento fallido conserva el último modelo y resultado exitosos; sólo se publican artefactos cuando la nueva ejecución termina correctamente. |

`ResultadoRegresion` contiene observaciones/predicciones/residuos de train y
prueba, métricas separadas, parámetros efectivos y modelo/preprocesador
ajustados. `metadatos` registra índices de train/test/validación, huella de datos,
semillas, opciones y versiones. Los residuos son `real - predicción`; los errores
se expresan en las unidades originales del objetivo (MSE en unidades al cuadrado).
El contenedor es `frozen`, pero sus objetos pandas y modelos internos son mutables:
deben tratarse como snapshots de lectura.

## Verificación del núcleo

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_regresion.py tests/test_framework.py
```

## Estimadores y catálogo

Todos utilizan `entrenar(algoritmo=..., **parametros)` y el mismo resultado.
Los argumentos del estimador se transmiten explícitamente; parámetros inválidos
producen un error sin reemplazar el último resultado exitoso.

| Identificador | Método pythonico | Configuraciones del benchmark |
|---|---|---|
| RLS | `lineal_simple` | Estándar, una predictora numérica |
| RLM | `lineal_multiple` | Estándar, dos o más predictoras |
| Lasso | `lasso` | `alpha=0.1`, `alpha=1` |
| LassoCV | `lasso_cv` | `alphas=(0.01, 0.1, 1, 10)` |
| Ridge | `ridge` | `alpha=0.1`, `alpha=1` |
| RidgeCV | `ridge_cv` | `alphas=(0.01, 0.1, 1, 10)` |
| SVR | `svr` | Kernel RBF, `C=1`, `C=10` |
| DT | `arbol_decision` | Profundidad libre, profundidad 5 |
| RF | `random_forest` | 100 árboles, profundidad libre o 5 |

`framework_ia.utils.configuraciones_experimento_regresion()` devuelve copias
independientes de los catorce candidatos, en ese orden. El catálogo de controles
`CONFIGURACIONES_REGRESION` comparte los mismos valores iniciales de los modelos.
Los identificadores de la tabla también están disponibles como aliases de método.
Fuera de RLS/RLM, los demás modelos admiten una o más predictoras.

### Validación interna y reproducibilidad

LassoCV y RidgeCV son los estimadores nativos de scikit-learn. Ambos reciben
`KFold(n_splits=5, shuffle=True, random_state=semilla)`; RidgeCV usa explícitamente
`scoring="neg_mean_squared_error"`, y LassoCV selecciona por MSE interno. El
framework fija estos controles: pasar `cv` o `scoring` produce un error claro,
no una sustitución silenciosa. Se puede personalizar la cuadrícula `alphas`.
El entrenamiento necesita al menos cinco filas para esa búsqueda interna.

**Limitación:** el preprocesamiento se ajusta una vez sobre todo el train recibido
antes de la búsqueda interna de alpha. No se reajusta dentro de esos cinco
pliegues; sus puntuaciones no constituyen evidencia independiente de rendimiento.
La evaluación externa del benchmark debe ajustar el preprocesamiento de nuevo
en cada pliegue externo. Consultar [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
y [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html).

El resultado registra `alpha_elegido` y `cv_interna` (pliegues, mezcla, semilla,
criterio y limitación). `parametros["cv"]` es una descripción serializable del
splitter; el objeto sklearn original permanece en `resultado.modelo.cv`.
Los modelos aleatorios utilizan la semilla de entrenamiento; RF y LassoCV
usan un trabajo por defecto (`n_jobs=1`).

## Benchmark reproducible

```python
experimento = regresion.experimentar(feature_simple="x1")
print(experimento.tabla)
print(experimento.mejor_algoritmo)
# benchmark(...) es un alias de experimentar(...).
```

El benchmark compara los catorce candidatos y selecciona una configuración por
algoritmo con el menor RMSE de validación. Todos comparten las mismas filas y
pliegues. RLS usa sólo `feature_simple`; los demás conservan las predictoras
seleccionadas. Si se omite, RLS utiliza la primera predictora numérica real en
el orden de `features`, sin inspeccionar correlaciones ni prueba. El preset de
referencia puede indicar `bmi` explícitamente; el núcleo no conoce ese nombre.

| Estrategia | Selección y ajuste final |
|---|---|
| Automática (`estrategia=None`) | CV si no hay validación externa; holdout si la partición compartida incluye validación. |
| `cv` | KFold mezclado de cinco pliegues sobre train. Cada pliegue ajusta su propio preprocesamiento. El ganador se reajusta con train completo. Una validación externa se conserva reservada, sin incorporarla a train. |
| `holdout` | Reutiliza validación externa o reserva `validation_size=0.2` de train. No modifica prueba ni reajusta con train+validación. |

`test_size=0.25`, `random_state=42`, `folds=5` y preprocesamiento son opciones
comunes. El score es la media aritmética de los RMSE por pliegue; su desviación
usa `ddof=1` (cero para una única validación). La tabla también muestra MAE,
MSE y R² medios, métricas por pliegue, parámetros efectivos y errores. Los
empates conservan el primer candidato y algoritmo declarados. El catálogo se
puede limitar con `algoritmos` o extender con `configuraciones`, manteniendo el
formato `{"Ridge": [{"nombre": "alpha_2", "parametros": {"alpha": 2}}]}`.

Todas las decisiones de validación terminan antes de evaluar ganadores en
prueba. `ResultadoExperimentoRegresion` contiene `tabla`, `mejores`,
`mejor_algoritmo` y `metadatos`; registra índices de particiones/pliegues,
selecciones, versiones y fallos. Ninguna métrica de prueba decide el ranking.
Un candidato fallido no impide evaluar los demás. Datos o particiones inválidos
se rechazan antes del primer ajuste. Si no puede evaluarse en prueba el ganador
elegido por validación, el intento devuelve su evidencia parcial y conserva
conjuntamente `experimento`, modelo, preprocesador, resultado y R² activos
anteriores (o vacíos en una instancia nueva). Otros algoritmos evaluados siguen
disponibles en el DTO devuelto, pero no se promueven por el resultado de prueba.

## CLI y exportación compartida

```powershell
.venv\Scripts\python.exe -m scripts.ejecutar_lab03 datos.csv --target respuesta --features x1 x2 --feature-simple x1 --salida salida_lab03
```

La CLI lee bytes locales, registra su SHA-256 y opciones de lectura, y no
descarga ni transmite datos. Conserva las opciones de lectura de Lab02:
`--separador`, `--decimal`, `--encoding`, además de `--fuente` para registrar
procedencia. `--algoritmos`, `--estrategia`, `--folds`, `--validation-size`,
`--test-size`, `--semilla`, `--incluir-categoricas`, `--sin-escalar` y
`--sin-imputar` controlan el experimento. RMSE es siempre la métrica de selección.

`framework_ia.exportacion_lab03.construir_archivos_experimento(...)` devuelve
los mismos bytes UTF-8 para CLI e interfaz, sin entrenar ni escribir archivos:

- `configuracion.json`: argumentos, procedencia, hash, versiones y protocolo.
- `comparacion_validacion.csv`: todos los candidatos, errores y scores.
- `mejores_modelos.json`: parámetros, métricas train/test y metadatos por ganador.
- `<algoritmo>/configuracion.json`, `metricas.json`, `predicciones_train.csv`
  y `predicciones_test.csv`: snapshot individual. Las predicciones conservan el
  índice `fila` y columnas `real`, `prediccion`, `residuo`.

Para una ejecución individual, `construir_archivos_modelo(resultado, *,
argumentos, dataset, versiones=None)` devuelve esos cuatro archivos sin el
prefijo de algoritmo; no necesita fabricar un experimento. Ambos constructores
conservan las versiones registradas al entrenar por encima de valores posteriores.
`construir_zip_experimento(archivos)` sólo agrupa esos bytes en memoria para
descarga; no prepara la entrega académica final.

La CLI no sobrescribe carpetas no vacías. Código de salida: **0** para éxito
completo, **1** para evidencia parcial exportada y **2** para entrada inválida o
ausencia de ganadores; en estos últimos casos no crea la carpeta de salida.

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_regresion_experimentos.py tests/test_lab3_cli.py
```

## Interfaz Nexo IA

### Orientación y requisitos

El flujo es **Fuente → Configuración → Ejecución → Resultados**.
La aplicación muestra la etapa actual y la siguiente acción. **Contexto activo**
despliega target, predictores, porcentaje de prueba y semilla efectivos.
**Configuración** es única: elija explícitamente Clasificación o Regresión y
prepare objetivo, predictores y protocolo comunes. No se deduce la tarea del
tipo de columna. Cada modelo conserva sus hiperparámetros y su propio botón
**Entrenar modelo**; configurar o cambiar de tarea nunca entrena.
Los accesos no disponibles explican el motivo en su ayuda al pasar el cursor.
Configuración y los botones de ejecución muestran qué falta: por ejemplo, corregir
un target con nulos, activar codificación de predictores categóricos, seleccionar
más variables para RLM o disponer de suficientes filas para validación cruzada.
Datos y Configuración permanecen accesibles para resolverlo.

No es obligatorio pulsar **Aplicar preparación** ni **Calcular partición**:
el split puede generarse al ejecutar y la preparación supervisada se aprende
solo con train. EDA/clustering son independientes. No se exige confirmar una
configuración válida ni visitar previamente todas las pantallas.

Clasificación usa clases discretas; regresión predice cantidades. El framework
no convierte automáticamente una cantidad continua en clases. Los diagnósticos
de disponibilidad no entrenan: sólo el botón explícito lo hace. Resultados se
habilita cuando hay evidencia vigente de esa familia y se deshabilita al cambiar
su contexto. Un intento parcial sigue siendo consultable.

La partición externa opcional también está en Configuración y reutiliza su
porcentaje, semilla y estratificación. Cambiar tarea, objetivo, predictores o
preparación descarta una partición incompatible. Sin partición externa, cada
ejecución divide automáticamente con el protocolo común. La imputación y el
escalado supervisados se aprenden con train; las transformaciones globales de
Datos son opciones avanzadas para exploración/clustering y no son obligatorias.

### Limpiar antes de configurar

En **Datos y preparación**, excluya columnas, quite duplicados y seleccione
**Eliminar filas con nulos en estas columnas**. La selección comienza vacía:
una fila sólo se elimina si tiene un nulo en alguna columna elegida. Si quiere
descartar filas sin objetivo, seleccione también esa columna explícitamente.
El filtro no rellena valores ni decide cuál será el target. Revise el conteo
previsto y pulse **Aplicar preparación**; Configuración utilizará ese resultado.

La limpieza se recalcula desde el original, no sobre una versión ya filtrada.
Se guardan política y conteos de filas retiradas por nulos/duplicados, y se
conservan las opciones al navegar. Si se eliminarían todas las filas, se muestra
un error y permanecen los datos y la evidencia anteriores. Aplicar una limpieza
válida invalida modelos y particiones anteriores. **Restaurar dataset original**
recupera los datos sin modificar el archivo fuente.

**Imputar** no significa eliminar filas: calcula valores de relleno. Para
supervisado, deje desactivadas las transformaciones globales y active imputación
de predictores/escalado en Configuración para aprenderlos sólo con train. Si ya
aplicó una transformación global, desactívela y vuelva a aplicar sus opciones de
limpieza; no necesita descartar las exclusiones o filtros seleccionados.

En Python, `EDA.preparar_dataset(columnas_sin_nulos=["x", "respuesta"],
imputar_nulos=False)` ofrece el mismo filtro. `columnas_sin_nulos=None` o `[]`
no elimina filas por nulos; los valores predeterminados anteriores de la API
se conservan. `eda.preparacion` describe la última preparación correcta.

Abra `app.py` con Streamlit. La identidad conserva los colores, tipografía,
controles nativos y temas claro/oscuro del proyecto; el nombre visible es **Nexo IA**.

1. Aplique un CSV en **Fuente de datos**. No se exige un dataset específico.
2. En **Configuración**, seleccione Regresión, target numérico, predictores y el
   predictor de RLS. Revise nulos y variación: el target no se imputa ni se
   descartan sus filas automáticamente. La partición externa es opcional.
3. Abra uno de los nueve modelos. **Estándar** usa el catálogo; **Personalizada**
   expone parámetros propios de regresión. Pulse **Entrenar modelo**.
4. Para seleccionar modelos, abra **Experimentos Lab03** y pulse
   **Ejecutar experimento**. Automática usa validación externa cuando existe;
   de lo contrario, CV de cinco pliegues. La tabla y su RMSE preceden a prueba.
5. **Resultados de regresión** compara snapshots sin entrenar: prueba individual,
   validación del benchmark y prueba de sus ganadores, en secciones separadas.
   No repite diagnósticos: vuelva a cada modelo para sus gráficos y descargas
   individuales. El benchmark conserva sus exportaciones. Las rutas antiguas
   de comparación abren este único destino. Clasificación sigue el mismo esquema.
   Las descargas conservan argumentos y procedencia de la ejecución, no los
   controles modificados posteriormente.

Editar parámetros conserva la evidencia y señala ajustes pendientes. Cambiar
datos, target, predictores, predictor simple, semilla, partición o preparación
la invalida. Un reintento fallido conserva el último éxito del mismo contexto.
Si falla la evaluación final del ganador por validación, el intento parcial
queda visible sin sustituir el experimento exitoso anterior ni elegir por test.
La imputación/escalado global previo al split bloquea el ajuste: desactívelos y
reaplique la limpieza; configure estas operaciones en Configuración para
aprenderlas con train.

Las métricas RMSE/MAE/MSE/R² son numéricas; R² puede ser negativo. Los gráficos
contrastan reales y predicciones y muestran residuos `real − predicción`.
Las exportaciones ZIP son evidencia técnica, no una entrega académica final.

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_lab3_ui.py tests/test_theme_contrast.py
```

AppTest verifica comportamiento y estado, no revisión visual real en navegador.
La extensión usa el diseño local mantenido; no se subieron datos ni diseños a
servicios externos para alineación visual.
