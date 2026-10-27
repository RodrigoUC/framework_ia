# LAB 02 · Clasificación supervisada

Requisitos: enunciado EIF420O Lab02 «Métodos supervisados: clasificación»
(Universidad Nacional de Costa Rica). Entrega: 15 de octubre de 2026.

## Alcance

Los cinco algoritmos del laboratorio (KNN, árbol de decisión, Random Forest,
XGBoost y AdaBoost) están en la jerarquía del framework
`EDA → Supervisado → Clasificacion` (`framework_ia/modelos/supervisado/`).

- `Clasificacion.entrenar(...)`: entrena una configuración con train/test.
- `Clasificacion.experimentar(estrategia="cv", folds=5, ...)`: protocolo completo
  del laboratorio. Con `estrategia="holdout"` usa una única partición de
  validación (comparación rápida; no es la usada en el informe).
- `scripts/ejecutar_lab02.py`: el mismo protocolo desde la terminal, con
  exportación de toda la evidencia. Por omisión usa validación cruzada.
- Streamlit (`app.py`): vistas individuales por algoritmo y **Experimentos Lab02**,
  que ofrece las dos estrategias (validación cruzada por omisión).

## Protocolo

1. Prueba reservada: 25 % estratificado (semilla 42), usado una sola vez al final.
2. Validación cruzada estratificada de 5 pliegues sobre train.
3. Catálogo (`utils/configuracion_clasificacion.py`): por familia, el estándar
   (valores predeterminados de la biblioteca) y tres variantes de un solo cambio.

   | Familia | Variantes |
   | --- | --- |
   | KNN | `k_15`, `pesos_distancia`, `manhattan` |
   | DT | `profundidad_5`, `hoja_minima_10`, `clases_balanceadas` |
   | RF | `500_arboles`, `hoja_minima_5`, `clases_balanceadas` |
   | XGBoost | `tasa_0_05`, `profundidad_3`, `clases_balanceadas` |
   | AdaBoost | `200_estimadores`, `tasa_0_5`, `clases_balanceadas` |

   `clases_balanceadas` pondera cada observación de train de forma inversa a la
   frecuencia de su clase (`compute_sample_weight("balanced")`). KNN no admite
   pesos por observación, por eso su tercera variante es la distancia.
4. Gana en cada familia la configuración con mayor F1 macro medio. Después se
   evalúa esa ganadora con `SelectKBest(mutual_info_classif)` y
   `k = ceil(atributos transformados / 2)`, ajustado dentro de cada pliegue; se
   conserva la mejor de las dos.
5. El elegido se reajusta con todo train y se evalúa en prueba: matriz de
   confusión, métricas por clase, curva ROC/AUC (clase minoritaria de train como
   positiva) e importancia por permutación (10 repeticiones, sólo interpretación).
   En DT se exportan las reglas del árbol con umbrales en unidades originales.

Imputación (mediana/moda), one-hot, estandarización y selección de atributos
viven en `PreprocesadorSupervisado` y se ajustan sólo con datos de entrenamiento
de cada pliegue. No aplique imputación ni escalado global desde EDA antes de
clasificar: la interfaz lo detecta y pide restaurar el original.

## Datos

Ver [`data/lab02/README.md`](../../data/lab02/README.md): Potabilidad usa los 2011
casos completos del archivo de Kaggle; Diabetes usa 390 pacientes con la clase
`glyhb > 7` y 17 predictoras.

## Archivos generados por experimento

- `configuracion.json`: argumentos, SHA-256 del CSV, versiones, índices de train,
  test y de cada pliegue, distribución de clases y atributos seleccionados.
- `comparacion_validacion.csv`: todas las configuraciones con media, desviación y
  F1 de cada pliegue.
- `mejores_modelos.json`: hiperparámetros efectivos, métricas de entrenamiento y
  de prueba, ROC y árbol del ganador de cada familia.
- `ranking_atributos.csv`: información mutua de cada atributo transformado.
- Por algoritmo: `matriz_confusion_test.csv`, `predicciones_test.csv`,
  `roc_test.csv`, `importancia_permutacion.csv` e `importancia_modelo.csv`
  (familias basadas en árboles). `DT/arbol.txt`: reglas completas.

## Verificación

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Las pruebas usan datos sintéticos y verifican el software (ausencia de fuga,
partición, CV, exportación, informe y empaquetado); no sustituyen los resultados
reales de `salida_lab02/`. Ver [VERIFICACION.md](VERIFICACION.md),
[INFORME.md](INFORME.md) y [LEEME_ENTREGA.md](LEEME_ENTREGA.md).
