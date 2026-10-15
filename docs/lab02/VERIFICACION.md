# Verificación de Lab02

## Datos y protocolo

- Diabetes original: 403 observaciones y 19 columnas; copia íntegra con SHA-256.
- Etiquetas: `Diabetes` si `glyhb > 7` y `No diabetes` si `glyhb <= 7`, según
  el criterio histórico de [HBiostat](https://hbiostat.org/data/repo/diabetes).
- Se excluyen las 13 observaciones sin `glyhb`, sin imputar etiquetas:
  390 casos, 60 `Diabetes` y 330 `No diabetes`.
- Se utilizan 17 predictoras nativas, sin `id` ni `glyhb`. Las categóricas son
  `location`, `gender` y `frame`. No se calculan BMI ni nuevas razones.
- Los faltantes de predictoras se conservan hasta el ajuste con entrenamiento;
  `bp.2s` y `bp.2d` tienen 252 ausencias cada una en el preparado.
- `trazabilidad_diabetes.csv` relaciona las 390 filas preparadas con sus filas
  e IDs originales. Los índices de fila comienzan en cero.
- Potabilidad: 3276 observaciones, nueve predictoras y objetivo `Potability`.
- Semilla 42, particiones estratificadas y selección por F1 macro de validación.
  Imputación, codificación y escala se ajustan exclusivamente con entrenamiento.

## Evidencia experimental

Cada dataset contiene 15 candidatos exitosos, cero fallidos y cinco ganadores.
Cada familia incluye una configuración estándar y dos variantes. La selección
no utiliza el conjunto de prueba.

| Dataset | Entrenamiento | Validación | Prueba |
| --- | ---: | ---: | ---: |
| Potabilidad | 1965 | 492 | 819 |
| Diabetes | 233 | 59 | 98 |

Los ganadores de Diabetes utilizan 21 columnas transformadas a partir de las
17 predictoras nativas. Los índices de los tres subconjuntos son únicos y
disjuntos. Las métricas por clase, accuracy y F1 macro coinciden con las
matrices de confusión y las predicciones exportadas.

La reproducción del protocolo desde una extracción limpia obtuvo los mismos
15 candidatos exitosos y cinco ganadores de Diabetes. Los 12 archivos de
resultados coincidieron byte a byte; el JSON de configuración difirió sólo en
`argumentos.salida`. Particiones, métricas, versiones y datos coincidieron.

## Software e interfaz

Las pruebas cubren el umbral estricto, etiquetas ausentes, trazabilidad,
preparación repetible, rechazo de campos CSV adicionales, integridad del paquete
y configuración de Streamlit. Configurar o navegar no entrena modelos.

La interfaz se verificó en anchos de 1440 px con tema claro, 1024 px con tema
oscuro y 390 px con tema claro, sin errores JavaScript ni desborde horizontal.
La descarga no ejecuta otro entrenamiento. Los 12 archivos de resultados
exportados por la interfaz para Diabetes coincidieron con la CLI; también
coincidió el JSON de experimento.

## Informe e integridad de entrega

El generador comprueba procedencia, hashes, particiones disjuntas, selección
por validación y coherencia de las métricas antes de construir el informe.
La fuente LaTeX es autónoma, con tablas, figuras y referencias integradas.

El PDF se valida mediante compilación y revisión visual de todas sus páginas.
`informe/verificacion_pdf.json` registra sus huellas, las páginas revisadas y
el orden de las secciones. El proyecto Overleaf incluye la misma fuente y su
manifiesto de procedencia.

El ZIP contiene cuatro CSV y 13 archivos de evidencia por dataset.
`MANIFIESTO.json` registra el tamaño y SHA-256 de cada archivo incluido.
La verificación comprueba CRC, hashes, copia original de Diabetes y su
derivación reproducible. Los 14 archivos conservados de Potabilidad mantienen
sus bytes y huellas.
