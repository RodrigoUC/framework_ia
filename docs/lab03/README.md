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

Los experimentos, estimadores adicionales, exportación y vistas del Lab03 se
incorporan en las siguientes unidades de implementación.
