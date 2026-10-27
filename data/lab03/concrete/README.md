# Concrete Compressive Strength de UCI

Referencia académica de regresión de [UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength),
distribuida bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
El original contiene 1030 observaciones y nueve columnas. Descargue el archivo
original `Concrete_Data.xls` de esa fuente y conserve sus bytes intactos.

Atribución: Prof. I-Cheng Yeh. Publicación original: *Modeling of strength of high
performance concrete using artificial neural networks*, Cement and Concrete
Research, 28(12), 1797-1808 (1998). El archivo informativo original solicita
conservar ese aviso y la referencia; ambos se registran en la procedencia.

```powershell
.venv\Scripts\python.exe -m scripts.preparar_concrete_lab03 RUTA\Concrete_Data.xls
```

La preparación sólo lee un archivo local; no descarga, imputa ni escala. Requiere
`xlrd`, incluido en las dependencias del proyecto. `--salida` permite elegir otro
directorio. Genera localmente el original preservado `Concrete_Data.xls`, el
CSV UTF-8/LF `concrete.csv` y `procedencia.json`; los datos no se versionan.

| Columna canónica | Significado | Unidad |
|---|---|---|
| cement | Cemento | kg/m³ |
| slag | Escoria de alto horno | kg/m³ |
| fly_ash | Ceniza volante | kg/m³ |
| water | Agua | kg/m³ |
| superplasticizer | Superplastificante | kg/m³ |
| coarse_aggregate | Agregado grueso | kg/m³ |
| fine_aggregate | Agregado fino | kg/m³ |
| age | Edad | días |
| strength | Resistencia a compresión, objetivo | MPa |

Se validan las cabeceras originales, las 1030 filas y todos los valores numéricos
reales y finitos, sin ausencias. Sólo se eliminan filas idénticas en las nueve
columnas, conservando la primera aparición y el orden original. Dos observaciones
con las mismas predictoras y distinta resistencia se conservan. La procedencia
registra las posiciones eliminadas desde cero, sin contar la cabecera de Excel.
La cantidad efectiva de filas preparadas debe obtenerse de esa procedencia.

La procedencia separa SHA-256 del XLS original y del CSV materializado, conserva
el mapeo de cabeceras y unidades, la licencia, la regla de preparación y las
versiones de las bibliotecas. No contiene fechas variables ni rutas absolutas.
Repetir la preparación en el mismo entorno conserva contenido idéntico sin
reescribirlo; un conflicto en cualquier destino se detecta antes de escribir
otros archivos. Use un directorio nuevo para una fuente o entorno diferente.

Para el experimento, seleccione `strength` como objetivo y las ocho variables
restantes como predictoras. `cement` se fija como ejemplo pedagógico de regresión
lineal simple antes de evaluar; no se afirma que sea la mejor variable. Los
demás regresores utilizan las ocho predictoras. El escalado y la imputación
supervisados se aprenden sólo con entrenamiento.

```powershell
.venv\Scripts\python.exe -m scripts.ejecutar_lab03 data/lab03/concrete/concrete.csv --target strength --feature-simple cement --salida salida_lab03/concrete --fuente https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
```

La carga por **CSV local** y **Subir CSV** ya está disponible en la interfaz.
Registre la URL de procedencia al aplicar la fuente y conserve este sidecar con
la evidencia de la entrega; cargar un CSV no importa el sidecar automáticamente.
