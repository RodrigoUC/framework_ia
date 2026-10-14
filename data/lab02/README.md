# Datos de Lab02

Los CSV se conservan localmente y se incluyen en la entrega académica, pero no
se publican automáticamente desde Git. `procedencia.json` registra sus huellas.

| Archivo | Fuente | Objetivo | Lectura |
| --- | --- | --- | --- |
| `water_potability.csv` | [Water Quality, Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability) | `Potability` | Coma como separador; punto decimal |
| `diabetes.csv` | [Predict diabetes based on diagnostic measures, Houcem Benmansour](https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures) | `diabetes` | Coma como separador; coma decimal dentro de valores entre comillas |

Potabilidad contiene 3276 observaciones y nueve predictoras. Se conserva el
archivo completo: la imputación de `ph`, `Sulfate` y `Trihalomethanes` se aprende
únicamente con entrenamiento. El ejemplo de clase muestra 2011 filas completas;
esa versión preparada no es la entrada empleada en estos experimentos.

Diabetes contiene 390 observaciones, 60 de la clase `Diabetes` y 330 de
`No diabetes`. Se excluye `patient_number`, que es un identificador, y se
conservan las otras 14 predictoras, incluida `gender` mediante codificación.
`chol_hdl_ratio`, `bmi` y `waist_hip_ratio` son variables numéricas con coma
decimal, por lo que no deben tratarse como categorías.

El esquema coincide con el caso del curso después de traducir los nombres y
excluir el ID; no se certifica igualdad completa con `diabetes_V2.csv` del
profesor. No se utiliza `Diabetes_prediction.csv` de 1000 registros.

Para reproducir:

```powershell
python -m scripts.ejecutar_lab02 data/lab02/water_potability.csv --target Potability --salida salida_lab02/potabilidad --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
python -m scripts.ejecutar_lab02 data/lab02/diabetes.csv --target diabetes --features cholesterol glucose hdl_chol chol_hdl_ratio age gender height weight bmi systolic_bp diastolic_bp waist hip waist_hip_ratio --decimal ',' --incluir-categoricas --salida salida_lab02/diabetes --semilla 42 --fuente https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures
```

Cada ejecución requiere una carpeta de salida nueva. No impute ni escale los
datos globalmente antes de estos comandos.
