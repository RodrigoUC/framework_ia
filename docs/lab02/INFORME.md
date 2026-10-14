# Informe de LAB 02 y proyecto Overleaf

`scripts/generar_informe_lab02.py` redacta las secciones experimentales a partir
de las exportaciones reales de `scripts.ejecutar_lab02`. No entrena ni modifica
los CSV. El texto de fundamentos y la bibliografía primaria se mantienen en
`fundamentos.tex` y `referencias.tex`; se incorporan al documento resultante.

La estructura visual adapta el estilo del informe del proyecto: `article`,
11 puntos, español, A4 y márgenes de 2.5 cm. No está acreditada como la plantilla
oficial del profesor. No se trasladan cifras ni conclusiones de LAB 01.

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

Desde la raíz, con los integrantes expresamente confirmados para este trabajo:

```powershell
python -m scripts.generar_informe_lab02 salida_lab02 --salida output/lab02/informe --procedencia data/lab02/procedencia.json --autor "Dylan Elizondo Alvarado (504610652)" --autor "Joseph Garcia Montero (402680547)" --autor "Rodrigo Ureña Castillo (118910482)" --autor "Mariana Ureña Padilla (119180324)"
```

Sin `--autor`, el documento marca la portada como pendiente. Si se permite una
ejecución incompleta con `--permitir-parcial`, el documento lo señala de forma
visible; no equivale al laboratorio completo.

## Evidencia verificada

Antes de guardar el informe, el generador comprueba:

- SHA-256 válido de cada entrada y coherencia con `procedencia.json`.
- Índices únicos y disjuntos en train, validación y test; tamaños compatibles.
- Preparación aprendida sólo con train y test excluido de la selección.
- Existencia de las cinco familias, estándar y variantes, y un ganador por
  familia que maximiza la métrica de validación.
- Coherencia entre selección, JSON de ganadores y evaluación final.
- Conteos no negativos, tamaños de test, accuracy, métricas por clase y F1
  macro consistentes con las matrices.

Una incoherencia detiene la generación; no se inventan ni corrigen cifras para
hacer coincidir un informe. Los candidatos fallidos se describen como tales.
El manifiesto registra hashes de los JSON/CSV utilizados y del TEX final.

## Salidas y compilación

- `main.tex`: documento autónomo, con tablas, figuras PGFPlots y referencias
  integradas mediante `thebibliography`.
- `manifest_informe.json`: autores, fecha y trazabilidad de fuentes.
- `README.md`: instrucciones breves para el proyecto de Overleaf.
- `proyecto_overleaf.zip`: esos tres archivos listos para importar en Overleaf.

En Overleaf, importar el ZIP y compilar `main.tex` con **pdfLaTeX**. No requiere
imágenes, archivos CSV externos ni ejecución de BibTeX. La compilación debe
verificarse y las páginas deben inspeccionarse antes de entregar un PDF.
`main.pdf` se añade sólo después de una compilación real exitosa.

Las predicciones completas y las matrices permanecen en la carpeta de
resultados; el proyecto de Overleaf no las reemplaza. El ZIP académico de
entrega debe incluir además código, datos con procedencia y resultados.

El informe diferencia Water Quality completo de la muestra de 2011 filas
del material de clase, y el CSV público de Diabetes de los encabezados
adaptados del cuaderno. Esa diferencia se documenta sin afirmar igualdad
íntegra no probada. La guía contempla trabajo individual o en pareja; la
composición final del grupo debe verificarse con el profesor. El documento
conserva los cuatro autores confirmados por el usuario para este proyecto.
