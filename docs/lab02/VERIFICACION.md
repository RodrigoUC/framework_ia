# Verificación de Lab02 — entrada original de Diabetes

Este registro corresponde al cambio a `diabetes_original.csv` de 403 filas y
19 columnas. Las verificaciones y mediciones de una entrada distinta de
Diabetes no se trasladan a esta preparación.

## Protocolo que debe acreditarse

- Preservar los bytes y SHA-256 del original recibido.
- Derivar `Diabetes` cuando `glyhb > 7` y `No diabetes` cuando `glyhb <= 7`,
  conforme al criterio histórico documentado en
  [HBiostat](https://hbiostat.org/data/repo/diabetes).
- Excluir las 13 filas sin `glyhb`, sin imputar etiquetas: 390 observaciones,
  60 `Diabetes` y 330 `No diabetes`.
- Conservar 17 predictoras nativas, sin `id` ni `glyhb`; incluir las
  categóricas `location`, `gender` y `frame`. No construir BMI ni una nueva
  razón cintura/cadera.
- Preservar los faltantes de predictoras, incluidas las 252 ausencias de
  `bp.2s` y las 252 de `bp.2d`. Aprender imputación, codificación y escala
  exclusivamente con entrenamiento.
- Relacionar cada fila preparada con su fila e ID originales, usando índices
  desde cero y conservando el hash de `trazabilidad_diabetes.csv`.
- Ejecutar las cinco familias con estándar y variaciones, semilla 42 y
  selección por F1 macro de validación. Mantener test fuera de la selección.

Potabilidad conserva Water Quality completo: 3276 observaciones, nueve
predictoras y objetivo `Potability`; su protocolo no cambia con esta
preparación de Diabetes.

## Estado de la nueva evidencia

La ejecución nueva de Diabetes produjo 15 candidatos exitosos, cero fallidos
y cinco ganadores. Los tamaños efectivos son 233 filas de entrenamiento,
59 de validación y 98 de prueba. Los cinco ganadores utilizan 21 columnas
transformadas a partir de las 17 predictoras nativas. Estas cantidades
proceden de los JSON/CSV de la nueva ejecución.

El generador comprobó las métricas contra las matrices de prueba, la selección
por validación, los índices disjuntos y la metadata de derivación. La fuente
LaTeX pasó la compilación integrada y Tectonic 0.17 exportó el PDF de 25 páginas.
Se renderizaron y revisaron todas las páginas, además de ampliar la portada,
la procedencia y exclusiones de Diabetes, la figura comparativa, la discusión,
las referencias y los comandos del anexo. No se observaron recortes ni
superposiciones; el log no contiene avisos `Overfull` ni referencias pendientes.
Las advertencias `Underfull` no afectan la legibilidad.

El PDF final tiene SHA-256
`ffbe36b962a122381ca2560d8daebc935e26bc03604da3d1d2ec4efc159a85a7`.
La fuente tiene SHA-256
`5a1806e48e4add46f80b02fb444ac8d8ee503ec939bf241836a2e8f765176249`.
El registro `output/lab02/informe/verificacion_pdf.json` conserva las huellas,
las páginas revisadas y el orden de secciones comprobado. El ZIP de Overleaf
incluye ese registro y la misma fuente que produjo el PDF.

La interfaz se verificó con ejecución por lotes en anchos de 1440 px (tema
claro), 1024 px (oscuro) y 390 px (claro), sin errores JavaScript ni desborde
horizontal. El botón de descarga no volvió a entrenar. Los 12 archivos
exportados por la interfaz para Diabetes coincidieron byte a byte con la
ejecución CLI nueva; también coincidió el JSON de experimento. Se comprobó que
los 14 archivos y huellas conservados de Potabilidad no cambiaron.

La suite completa terminó con **133 pruebas y 55 subpruebas aprobadas**, sin
fallos, en Python 3.11 con las versiones fijadas. Incluye el umbral estricto,
etiquetas ausentes, trazabilidad, preparación repetible, rechazo de campos CSV
adicionales, integridad del paquete y el perfil de Streamlit. Configurar,
navegar o intentar accionar un botón deshabilitado no entrena modelos.

Se extrajo un paquete de revisión en una carpeta limpia: CRC correcto y los
112 archivos del manifiesto con tamaño y SHA-256 verificados. La preparación
se validó usando el código extraído y se cotejaron las 390 filas, sus IDs y
etiquetas. Desde esa carpeta se ejecutó de nuevo Diabetes: 15 configuraciones
exitosas, cero errores y cinco ganadores. Los 12 archivos de resultados
coincidieron byte a byte; la única diferencia del JSON de configuración fue
`argumentos.salida`. Particiones, métricas, versiones y datos coincidieron.

El paquete final conserva el mismo código, datos, resultados y archivos del
informe que el paquete reproducido. Sólo cambian la documentación de uso, este
registro y las huellas correspondientes del manifiesto. Se verifican de nuevo
CRC, todos los tamaños y hashes del paquete final y los 14 archivos conservados
de Potabilidad.

El generador del informe verifica hashes, particiones disjuntas, selección
por validación y coherencia de métricas con matrices, además de la metadata
de derivación de Diabetes. El PDF se valida mediante compilación real y
revisión de todas sus páginas. La comprobación vigente se guarda en
`informe/verificacion_pdf.json` dentro de la entrega.

La organización del informe adapta la de Lab01 por indicación de los
integrantes; su contenido y experimentos corresponden a Lab02. La plantilla
oficial y la autorización docente para cuatro integrantes no se han
acreditado con los materiales disponibles. El proyecto se prepara para
Overleaf; no se carga en Overleaf ni se entrega en Moodle automáticamente.
