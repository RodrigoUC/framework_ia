# Ejecutar y comprobar Lab01

Esta carpeta incluye el código, los datos y el informe de Lab01. Se puede copiar
fuera del repositorio principal: no necesita archivos de `C:\framework_ia`.
Necesita Python y las dependencias de `requirements.txt` en el equipo de destino.

## 1. Abrir la carpeta y preparar Python

Extraiga la carpeta completa del ZIP. Abra PowerShell **dentro de esa carpeta**,
donde se encuentran `requirements.txt`, `framework_ia` y `scripts`.
Todos los comandos siguientes usan rutas relativas a esa ubicación.

Se recomienda **Python 3.12**. Compruebe que está instalado:

```powershell
py -3.12 --version
```

Si no encuentra esa versión, instale Python 3.12 antes de continuar. El comando
`python` puede apuntar a otra versión; tener Python 3.14 no equivale a tener
instaladas las dependencias de este proyecto.

Cree un entorno local e instale las dependencias:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se utiliza directamente el Python del entorno; no es necesario activar scripts
ni modificar la política de ejecución de PowerShell. La instalación requiere
acceso a Internet si las dependencias no están disponibles en caché.

## 2. Ejecutar los experimentos

Primero puede comprobar los argumentos disponibles:

```powershell
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab01 --help
```

Ejecute ACP, K-Means, HAC, t-SNE y UMAP con los datos incluidos:

```powershell
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab01 data/ejemplo_analisis.csv --salida resultados --semilla 42
```

El archivo conserva su nombre `ejemplo_analisis.csv`; es el conjunto confirmado
para este laboratorio. Tiene 16 observaciones, cuatro variables numéricas
(`edad`, `ingresos`, `visitas`, `satisfaccion`) y la categoría `segmento`.
Los experimentos usan las variables numéricas y las escalan; `segmento` no se
usa para ajustar los métodos no supervisados.

El ejecutor guarda los artefactos en `resultados`. Una nueva ejecución con esa
ruta actualiza los archivos con el mismo nombre. Las configuraciones que no son
válidas para el tamaño de la muestra se documentan en los resultados; por
ejemplo, una perplexity de 30 no puede usarse con 16 observaciones.

## 3. Comprobar los resultados

```powershell
.\.venv\Scripts\python.exe -m scripts.verificar_lab01 --resultados resultados
Get-ChildItem .\resultados
Get-Content .\resultados\resumen_experimentos.json
```

El verificador debe terminar sin errores. Revise las comparaciones de los cinco
métodos y que el resumen no registre un error de UMAP. El mensaje
"Experimentos completados" por sí solo no demuestra que todos los métodos
hayan terminado correctamente.

La verificación comprueba también archivos obligatorios, cantidades de filas,
selecciones y huellas SHA256. No edite manualmente los CSV generados; si cambia
los datos o los parámetros, vuelva a ejecutar los experimentos.

| Contenido | Ubicación |
|---|---|
| Configuración, comparaciones y resumen | `resultados/` |
| Estadísticos exploratorios y matriz preparada | `resultados/` |
| Proyecciones, asignaciones, cargas/varianzas de ACP e información jerárquica de HAC | `resultados/` |
| Código de los experimentos y verificación | `scripts/` |
| Clase `Cluster` y métodos de agrupamiento/reducción | `framework_ia/modelos/no_supervisado/agrupamiento.py` |
| Informe LaTeX | `informe/plantilla_informe.tex` |
| Bibliografía reutilizable | `informe/referencias.bib` |

Si copia la carpeta a otra ubicación, abra PowerShell en la nueva raíz y repita
los mismos comandos. No copie un entorno `.venv` entre equipos: créelo en el
equipo donde vaya a ejecutar el código.

## 4. Abrir y compilar el informe

El archivo `.tex` contiene el código fuente del documento. Abra
`informe/plantilla_informe.tex` en un editor LaTeX, o súbalo a un proyecto de
Overleaf y compílelo con **pdfLaTeX** para obtener y descargar el PDF.

Las tablas y los gráficos están incorporados en el `.tex`; no necesitan imágenes
externas. El documento incluye la bibliografía en su propio código para poder
compilarse como archivo independiente. `referencias.bib` se conserva como
respaldo de las referencias y para reutilización con BibTeX; no hace falta
cambiar la bibliografía del documento para compilarlo.

Ejecutar los experimentos **no recompila ni modifica el informe**. Si cambia el
dataset o los parámetros, revise también las tablas, los gráficos y el análisis
del documento antes de volver a compilarlo. La compilación en Overleaf debe
realizarse allí; tener el archivo preparado no significa que ya se haya hecho.

### Estado de la validación local

La ejecución y la copia fuera del repositorio pasaron sus comprobaciones;
el registro está en `resultados/validacion_entrega.json`. El informe pasó
controles estáticos de cifras, autores, citas y estructura LaTeX.

El compilador integrado no pudo iniciar y devolvió
`Unable to find standard directories for platform`. Por eso todavía no se
ha verificado la apariencia del PDF ni se ha exportado uno. Compile el mismo
archivo en Overleaf y revise el PDF antes de entregarlo.

## 5. Preparar el ZIP

Conserve `data`, `framework_ia`, `scripts`, `resultados`, `informe` y los archivos
de documentación y configuración de la raíz. Tras compilar el informe, agregue
el PDF descargado a `informe`.

No incluya `.venv`, `__pycache__`, `.git` ni archivos temporales de compilación.
El `.gitignore` ayuda al control de versiones, pero **no excluye archivos al
comprimir desde el Explorador de Windows**: compruebe el contenido del ZIP.
No se requiere ejecutar una interfaz web para reproducir estos experimentos.
