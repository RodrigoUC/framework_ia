# Lab03: regresión y benchmarking

El núcleo de regresión conserva la jerarquía `Supervisado → Regresion` y separa
los modelos de la interfaz. `Progresion` sigue siendo un alias compatible.

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

Los experimentos, exportación y vistas del Lab03 se incorporan en las siguientes
unidades de implementación.
