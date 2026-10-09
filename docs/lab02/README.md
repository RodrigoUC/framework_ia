# LAB 02 · Clasificación supervisada

Fuente de requisitos: [`guia_clasificacion.pdf`](../lab01/guia_clasificacion.pdf),
«Métodos supervisados: clasificación», Universidad Nacional de Costa Rica.
La guía permanece en su ubicación original; no es la guía de regresión.

## Alcance y jerarquía

El laboratorio incorpora KNN, Árbol de Decisión, Random Forest, XGBoost y
AdaBoost en `Supervisado → Clasificacion`. Conserva RF/NR y la organización
`datos/`, `modelos/supervisado/`, `resultados.py`, `visualizacion.py`, `ui/`.
ACP, t-SNE y UMAP siguen siendo métodos de `Cluster`, como pide el refactor
del profesor. Regresión no forma parte de este cambio.

En Streamlit, las seis vistas individuales (incluida Naive Bayes como
compatibilidad, fuera de las cinco familias experimentales) permiten entrenar
explícitamente. Antes del entrenamiento se presenta el balance del objetivo.
**Comparar configuraciones** permite usar variantes predefinidas, solo el
estándar o variantes JSON; presenta los candidatos antes de ejecutar.
**Resultados de clasificación** reúne, sin volver a entrenar, los modelos
individuales y los experimentos. La tabla de métricas de prueba individuales
es exploratoria: la selección de variantes se hace con validación, no con test.

- `Clasificacion.entrenar(...)`: una configuración, con train/test separado.
- `Clasificacion.experimentar(...)`: configuración estándar y variantes por
  algoritmo, selección por F1 macro en validación y evaluación final en test.
- `ResultadoExperimentoClasificacion`: tabla de comparación, mejores modelos
  por algoritmo y metadatos, sin depender de Streamlit.
- `python -m scripts.ejecutar_lab02`: el mismo flujo desde la terminal.

## Protocolo experimental

1. Leer el CSV original y seleccionar explícitamente objetivo y predictoras.
2. Separar train, validación y test de forma reproducible y estratificada.
3. Ajustar imputación, codificación y estandarización **sólo con train**.
4. Entrenar cada configuración con la misma partición; comparar en validación.
5. Seleccionar un ganador por algoritmo mediante F1 macro. Registrar los
   hiperparámetros efectivos y la semilla; no elegir por el resultado de test.
6. Evaluar los ganadores en test y analizar errores y distribución por clase.

El test no debe convertirse en validación por repetir ajustes tras inspeccionar
sus resultados. Las métricas no tienen significado clínico ni certifican
seguridad de agua. Un único holdout tiene incertidumbre: el informe debe
reconocer limitaciones, desbalance y posible cambio de distribución.

No aplicar imputación/escalado global desde EDA antes de clasificar: eso usa
información de test. La UI del LAB 02 detecta esa preparación y pide restaurar
el original; sus controles propios delegan las transformaciones al modelo.
La exclusión manual de columnas requiere justificación: nunca use la variable
objetivo como predictora ni incluya identificadores que revelen la etiqueta.

## CSV de Kaggle y procedencia

Los datos deben permanecer locales; no se incluyen en el repositorio.
Potabilidad disponible públicamente: [Water Quality, Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability),
versión 3, CC0. `water_potability.csv` contiene 3276 filas, 9 predictoras y
objetivo `Potability` (0/1); tiene nulos en `ph`, `Sulfate` y `Trihalomethanes`.
La coincidencia con el archivo específico del profesor debe comprobarse.

«Diabetes» no identifica de forma unívoca un dataset. Use el CSV y la columna
objetivo de la fuente elegida; no transforme el objetivo continuo de
`sklearn.datasets.load_diabetes` en una clase mediante un umbral arbitrario.
El cargador no descarga archivos ni presupone columnas específicas.
No elimine ceros como si fueran nulos sin comprobar su significado en la ficha
y documentar esa decisión (por ejemplo, cero embarazos puede ser válido).

## Ejecución

Desde la raíz del repositorio (Python 3.11+ y Streamlit 1.65+):

```bash
python -m pip install -r requirements.txt
python -m scripts.ejecutar_lab02 /ruta/water_potability.csv \
  --target Potability --salida salida_lab02/potabilidad \
  --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
```

Para un CSV con variables categóricas, agregue `--incluir-categoricas`.
`--features columna_1 columna_2` permite estudiar un subconjunto explícito.
`--algoritmos KNN DT` sirve para un ensayo rápido; el laboratorio completo
requiere las cinco familias. `--test-size` y `--validation-size` configuran
la partición. Validación es una fracción del train restante tras separar test: con
los valores predeterminados se obtiene 60% train, 15% validación y 25% test.
Los ganadores conservan su ajuste con train, sin reajuste con validación.
Consulte `--help` y los metadatos exportados para los tamaños exactos.

Use una carpeta de salida nueva: el script rechaza directorios no vacíos para
no sobrescribir evidencia anterior. Fije versiones de dependencias al repetir
una ejecución histórica; una semilla por sí sola no garantiza igualdad entre
versiones o plataformas.

### Archivos generados

- `configuracion.json`: argumentos, hash SHA-256 del CSV, referencia de origen,
  versiones de bibliotecas y metadatos de la partición/experimento.
- `comparacion_validacion.csv`: configuraciones y desempeño de validación.
- `mejores_modelos.json`: métricas de test, parámetros y trazabilidad del ganador
  de cada algoritmo.
- Carpetas por algoritmo con matriz de confusión y predicciones de test.

Las predicciones pueden contener información derivada del dataset. Son salidas
locales: no publicarlas automáticamente, especialmente con datos médicos o
material cuya licencia limite redistribución.

## Guía para interpretar, no conclusiones predeterminadas

- **KNN:** votación de vecinos; analizar número de vecinos, distancia y efecto
  del escalado. Muchos atributos irrelevantes pueden deteriorar la distancia.
- **Árbol:** particiones sucesivas; comparar profundidad y mínimos por hoja,
  interpretabilidad y sobreajuste.
- **Random Forest:** ensamble de árboles sobre muestras/atributos variados;
  comparar número de árboles, profundidad y balance entre clases.
- **XGBoost:** árboles aditivos que corrigen errores mediante optimización de
  gradiente; discutir tasa de aprendizaje, profundidad y número de árboles.
- **AdaBoost:** combinación secuencial de aprendices débiles que enfatiza
  observaciones difíciles; analizar estimadores, aprendizaje y sensibilidad
  a ruido/atípicos.

Para cada familia explique por qué la variante ganadora supera o no al estándar,
revise precisión/recall/F1 por clase y falsos positivos/negativos. Una exactitud
alta puede esconder una clase minoritaria mal detectada. No interprete una
importancia de variable como causalidad ni como recomendación médica.

Referencias técnicas primarias para el informe:
[clasificación de sklearn](https://scikit-learn.org/stable/supervised_learning.html),
[prevención de fuga de información](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage),
[XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
Las definiciones y decisiones finales deben comprenderse, desarrollarse y
referenciarse por los integrantes en la plantilla LaTeX del profesor.

## Verificación y entrega

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Las pruebas de software usan fixtures sintéticos; no son experimentos académicos
sobre los CSV del profesor. Un resultado de prueba nunca sustituye la tabla real.
La verificación headless no certifica contraste visual, navegación por teclado
ni comportamiento en distintos tamaños de pantalla.
La entrega final exige informe con definiciones, experimentación y análisis,
plantilla oficial en LaTeX/Overleaf y ZIP con IDs de integrantes. Esta
implementación no presenta un informe final ni entrega archivos en Moodle.
