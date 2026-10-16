# Referencia de regresión de Lab03

Ejecute `python -m scripts.preparar_diabetes_lab03` desde la raíz. Se crean
`diabetes_regresion.csv` y `procedencia.json` localmente; no se versionan ni
transmiten datos. `--salida` permite otro directorio. El cargador usa los
archivos incluidos en la instalación de scikit-learn, sin red.

| Aspecto | Valor |
|---|---|
| Cargador | `load_diabetes(as_frame=True, scaled=False)` |
| Tamaño | 442 filas, diez predictoras y `target` |
| Objetivo | Progresión cuantitativa al año, no clase diagnóstica |
| Predictoras | age, sex, bmi, bp, s1, s2, s3, s4, s5, s6 |
| Ejemplo RLS | bmi; sólo el preset lo elige, no el modelo |
| Preparación | Sin imputación ni escalado; el entrenamiento los aprende sólo de train |
| Huella | SHA-256 del CSV materializado UTF-8/LF/sin índice |

La procedencia registra función/opciones de carga, esquema, serialización,
preset y versiones de Python/numpy/pandas/sklearn. Una nueva versión puede
cambiar la serialización o procedencia: use otro directorio en vez de
sobrescribir evidencia previa. Repetir en el mismo entorno no reescribe archivos
idénticos; cualquier conflicto se detecta antes de escribir alguno.

Esta referencia es distinta de **Diabetes de HBiostat, Lab02**, cuyo objetivo
binario se deriva de `glyhb > 7`. No intercambie sus perfiles, filas ni métricas.
Consulte la [documentación del cargador](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
y la [guía de ejecución](../../docs/lab03/README.md).
