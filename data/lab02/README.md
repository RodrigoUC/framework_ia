# Datos de Lab02

Los CSV se incluyen en la entrega académica. Si el repositorio Git no los
contiene, los originales se descargan de las fuentes indicadas y los preparados
se regeneran con los comandos del final. `procedencia.json` registra fuente,
reglas de preparación y SHA-256 de cada archivo.

| Archivo | Qué es | Filas |
| --- | --- | --- |
| `water_potability.csv` | Original de [Water Quality, Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability), sin modificar | 3276 |
| `potabilidad.csv` | Casos completos del original; es el archivo usado en los experimentos | 2011 |
| `diabetes_original.csv` | Original documentado en [HBiostat: Diabetes Dataset](https://hbiostat.org/data/repo/diabetes), sin modificar | 403 |
| `diabetes.csv` | Preparado con la clase `diabetes`; es el archivo usado en los experimentos | 390 |
| `trazabilidad_diabetes.csv` | Relación fila preparada → fila e `id` originales | 390 |

Todos usan coma como separador, punto decimal y UTF-8.

## Potabilidad

El original tiene valores ausentes en `ph` (491), `Sulfate` (781) y
`Trihalomethanes` (162). La versión de clase del curso usa los 2011 casos
completos, y eso es lo que produce `scripts.preparar_potabilidad_lab02`: elimina
las filas con algún faltante, no imputa nada y conserva el orden original.
Resultado: 1200 no potables (0) y 811 potables (1).

## Diabetes

El original no trae una clase. Se usa el criterio histórico de la documentación
del dataset: `glyhb > 7` es `Diabetes`, `glyhb <= 7` es `No diabetes` y las 13
filas sin `glyhb` se excluyen (no se inventa una etiqueta). Quedan 390 pacientes:
60 con diabetes y 330 sin diabetes.

Se retiran `id` (identificador) y `glyhb` (define la clase; usarla sería fuga
directa). Quedan 17 predictoras:

| Tipo | Columnas |
| --- | --- |
| Categóricas | `location`, `gender`, `frame` |
| Numéricas | `chol`, `stab.glu`, `hdl`, `ratio`, `age`, `height`, `weight`, `bp.1s`, `bp.1d`, `bp.2s`, `bp.2d`, `waist`, `hip`, `time.ppn` |

Los faltantes de las predictoras se conservan (por ejemplo, 252 en `bp.2s` y
`bp.2d`). La imputación, la codificación one-hot y el escalado se aprenden sólo
con entrenamiento dentro de cada experimento.

## Regenerar

```powershell
python -m scripts.preparar_potabilidad_lab02 data/lab02/water_potability.csv
python -m scripts.preparar_diabetes_lab02 data/lab02/diabetes_original.csv
```

Ambos son deterministas: repetirlos con el mismo original produce los mismos
bytes, y se niegan a reemplazar un CSV existente con contenido distinto.
