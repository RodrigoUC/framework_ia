# Datos de Lab02

Los CSV se conservan localmente y se incluyen en la entrega académica, pero no
se publican automáticamente desde Git. `procedencia.json` registra sus huellas.

| Archivo | Fuente | Objetivo | Lectura |
| --- | --- | --- | --- |
| `water_potability.csv` | [Water Quality, Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability) | `Potability` | Coma como separador; punto decimal |
| `diabetes_original.csv` | Archivo original recibido, documentado en [HBiostat: Diabetes Dataset](https://hbiostat.org/data/repo/diabetes) | `glyhb`, antes de derivar la clase | Coma como separador; punto decimal; UTF-8 |
| `diabetes.csv` | Preparación reproducible del original anterior | `diabetes`, derivado con la regla documentada | Coma como separador; punto decimal; UTF-8 |

Potabilidad contiene 3276 observaciones y nueve predictoras. Se conserva el
archivo completo: la imputación de `ph`, `Sulfate` y `Trihalomethanes` se aprende
únicamente con entrenamiento. El ejemplo de clase muestra 2011 filas completas;
esa versión preparada no es la entrada empleada en estos experimentos.

## Preparación de Diabetes

El original tiene 403 filas y 19 columnas. No contiene una clase categórica
de diabetes. La documentación primaria describe el criterio histórico
`glyhb > 7.0`; con ese respaldo y la decisión expresa del trabajo se deriva:

- `glyhb > 7`: `Diabetes`.
- `glyhb <= 7`: `No diabetes`.
- `glyhb` ausente: excluir la fila; no imputar una etiqueta desconocida.

Son 13 exclusiones y 390 filas etiquetadas: 60 `Diabetes` y 330 `No diabetes`.
El criterio se documenta como regla experimental histórica del conjunto;
no se presenta como diagnóstico clínico actual ni como una regla explícita
del enunciado. No se certifica igualdad íntegra con un archivo docente adaptado.

Se retiran `id` y `glyhb` del archivo preparado: el primero es identificador;
el segundo define la respuesta y sería fuga directa si fuera predictor.
Se conservan las 17 predictoras nativas, sin construir BMI ni una nueva
razón cintura/cadera. `ratio` ya existe en la entrada. El preparado contiene
esas 17 predictoras y la nueva columna `diabetes` (390 × 18).

| Tipo | Columnas |
| --- | --- |
| Categóricas | `location`, `gender`, `frame` |
| Numéricas | `chol`, `stab.glu`, `hdl`, `ratio`, `age`, `height`, `weight`, `bp.1s`, `bp.1d`, `bp.2s`, `bp.2d`, `waist`, `hip`, `time.ppn` |

La preparación no elimina filas por faltantes de predictoras ni las imputa.
`bp.2s` y `bp.2d` conservan 252 ausencias cada una entre las 390 filas.
La imputación numérica por mediana, la categórica por moda, la codificación
one-hot y el escalado se aprenden únicamente con entrenamiento durante cada
experimento. No se usa un filtrado global de casos completos.

`trazabilidad_diabetes.csv` relaciona `fila_preparada`, `fila_original` e
`id_original`, con índices de fila desde cero. Los índices de partición
exportados por la CLI corresponden al preparado; esta tabla permite llegar
a la fila original. `procedencia.json` conserva la regla, las 13 exclusiones,
las rutas y SHA-256 de original, preparado y correspondencia.

Para reproducir:

```powershell
python -m scripts.ejecutar_lab02 data/lab02/water_potability.csv --target Potability --salida salida_lab02/potabilidad --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
python -m scripts.preparar_diabetes_lab02 data/lab02/diabetes_original.csv --salida data/lab02
python -m scripts.ejecutar_lab02 data/lab02/diabetes.csv --target diabetes --features chol stab.glu hdl ratio location age gender height weight frame bp.1s bp.1d bp.2s bp.2d waist hip time.ppn --incluir-categoricas --salida salida_lab02/diabetes --semilla 42 --fuente https://hbiostat.org/data/repo/diabetes
```

Cada experimento requiere una carpeta de salida nueva. El preparador admite
repetición idéntica; si cambiarían los datos, exige una carpeta limpia para no
mezclar entradas. No impute ni escale globalmente antes de estos comandos.
