# Informe de LAB 02 y proyecto Overleaf

`scripts/generar_informe_lab02.py` redacta las secciones experimentales a partir
de las exportaciones reales de `scripts.ejecutar_lab02`. No entrena ni modifica
los CSV. El texto de fundamentos y la bibliografía primaria se mantienen en
`fundamentos.tex` y `referencias.tex`; se incorporan al documento resultante.

El documento utiliza la clase `article`, 11 puntos, español, papel A4 y márgenes
de 2.5 cm.

## Generación

Se requieren ambas carpetas experimentales completas:

```text
salida_lab02/
  potabilidad/
    configuracion.json
    comparacion_validacion.csv
    mejores_modelos.json
    KNN/matriz_confusion_test.csv
    DT/matriz_confusion_test.csv
    RF/matriz_confusion_test.csv
    XGBoost/matriz_confusion_test.csv
    AdaBoost/matriz_confusion_test.csv
  diabetes/
    (misma estructura)
```

Desde la raíz del proyecto:

```powershell
python -m scripts.generar_informe_lab02 salida_lab02 --salida output/lab02/informe --procedencia data/lab02/procedencia.json --autor "Dylan Elizondo Alvarado (504610652)" --autor "Joseph Garcia Montero (402680547)" --autor "Rodrigo Ureña Castillo (118910482)" --autor "Mariana Ureña Padilla (119180324)"
```

Sin `--autor`, el documento indica autoría no especificada. La opción
`--permitir-parcial` habilita documentos con resultados incompletos,
identificados como parciales.

## Evidencia verificada

Antes de guardar el informe, el generador comprueba:

- SHA-256 válido de cada entrada y coherencia con `procedencia.json`.
- En Diabetes, procedencia del original 403 × 19, regla `glyhb > 7`, 13
  objetivos ausentes excluidos, preparado de 390 filas, 17 predictoras sin
  `id`/`glyhb` y correspondencia de filas con SHA-256. Un resultado de otra
  preparación se rechaza antes de escribir el informe.
- Índices únicos y disjuntos en train, validación y test; tamaños compatibles.
- Preparación aprendida sólo con train y test excluido de la selección.
- Existencia de las cinco familias, estándar y variantes, y un ganador por
  familia que maximiza la métrica de validación.
- Coherencia entre selección, JSON de ganadores y evaluación final.
- Conteos no negativos, tamaños de test, accuracy, métricas por clase y F1
  macro consistentes con las matrices.

Las incoherencias detienen la generación. Los candidatos fallidos conservan
su estado y el error correspondiente.
El manifiesto registra hashes de los JSON/CSV utilizados y del TEX final.

## Salidas y compilación

- `main.tex`: documento autónomo, con tablas, figuras PGFPlots y referencias
  integradas mediante `thebibliography`.
- `manifest_informe.json`: autores, fecha y trazabilidad de fuentes.
- `README.md`: instrucciones breves para el proyecto de Overleaf.
- `proyecto_overleaf.zip`: esos tres archivos listos para importar en Overleaf;
  tras validar el PDF final se agrega `verificacion_pdf.json`, con sus huellas
  y el registro de revisión visual.

En Overleaf, importar el ZIP y compilar `main.tex` con **pdfLaTeX**. No requiere
imágenes, archivos CSV externos ni ejecución de BibTeX. La compilación debe
verificarse y las páginas deben inspeccionarse antes de entregar un PDF.
`main.pdf` se añade sólo después de una compilación real exitosa.

Las predicciones completas y las matrices permanecen en la carpeta de
resultados; el proyecto de Overleaf no las reemplaza. El ZIP académico de
entrega debe incluir además código, datos con procedencia y resultados.

El informe diferencia Water Quality completo de la muestra de 2011 filas
del material de clase. En Diabetes documenta la preparación del archivo
original y el criterio histórico respaldado por HBiostat:
`glyhb > 7` produce `Diabetes`, `glyhb <= 7` produce `No diabetes`; 13 filas
sin objetivo se excluyen sin imputar. Conserva 17 predictoras nativas,
incluidas `location`, `gender` y `frame`, con faltantes preservados hasta el
ajuste exclusivo de train.

El anexo incluye el comando de `scripts.preparar_diabetes_lab02`, los hashes
del original y del mapa `trazabilidad_diabetes.csv`, y la relación de índices
preparados con los originales. La estructura es introducción, fundamentos,
metodología, resultados por dataset, discusión y limitaciones, conclusiones, referencias
y anexo de reproducción. La identificación de fuente y esquema no acredita
identidad completa con otras adaptaciones del dataset.
