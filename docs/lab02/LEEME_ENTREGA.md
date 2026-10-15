# Entrega Lab02

El ZIP contiene cuatro CSV: Potabilidad, Diabetes original, Diabetes preparado y
su trazabilidad. Incluye `data/lab02/procedencia.json`, el framework, la aplicación,
los ejecutores, pruebas, resultados de ambos datasets y el informe PDF con su
fuente LaTeX. `MANIFIESTO.json` permite comprobar el SHA-256 y tamaño de cada
archivo; no incorpora rutas absolutas ni fechas variables.

Para abrir la aplicación tras extraer el ZIP, abra PowerShell en la carpeta extraída:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lab02-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La configuración de tema Atlas se conserva en `.streamlit/config.toml`. Para
Diabetes, cargue `data/lab02/diabetes.csv` y pulse **Aplicar configuración Diabetes
Lab02** en Clasificación o Experimentos Lab02. El botón selecciona el objetivo,
las 17 predictoras y el protocolo experimental; entrenar requiere pulsar después
**Ejecutar experimentos Lab02**. El CSV se lee con separador coma y punto decimal,
como Potabilidad. Si carga el original, la interfaz le indica usar el preparado.

La entrada original de Diabetes tiene 403 filas y 19 columnas. La preparación
asigna `Diabetes` cuando `glyhb > 7` y `No diabetes` cuando `glyhb <= 7`; excluye
únicamente las 13 filas sin `glyhb`. El preparado tiene 390 filas, 17 predictoras
y la clase. `id` y `glyhb` quedan fuera de las predictoras; `location`, `gender`
y `frame` se codifican dentro del pipeline. Los faltantes en las predictoras se
conservan: su imputación se aprende con entrenamiento.

La regla de clase y la procedencia se registran en `procedencia.json`, que
contiene huellas separadas del original, del preparado y de la trazabilidad.
`trazabilidad_diabetes.csv` permite relacionar las filas originales con su
preparación. Para verificar y regenerar esos archivos desde el original incluido:

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab02 data/lab02/diabetes_original.csv --salida data/lab02
```

El preparador acepta repetir la operación con el mismo contenido y rechaza
reemplazar CSV existentes diferentes. La huella que registra cada experimento
corresponde al CSV preparado que realmente leyó.

Para repetir los experimentos, use carpetas de salida nuevas:

```powershell
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 data/lab02/water_potability.csv --target Potability --salida salida_lab02_reproducida/potabilidad --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 data/lab02/diabetes.csv --target diabetes --features chol stab.glu hdl ratio location age gender height weight frame bp.1s bp.1d bp.2s bp.2d waist hip time.ppn --incluir-categoricas --salida salida_lab02_reproducida/diabetes --semilla 42 --fuente https://hbiostat.org/data/repo/diabetes
```

Para preparar el paquete desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_entrega_lab02 --ids 504610652 402680547 118910482 119180324 --informe output/lab02/informe/main.pdf --fuentes-tex output/lab02/informe/main.tex
```

En una copia extraída del ZIP, use `--informe informe/main.pdf` y
`--fuentes-tex informe/main.tex`. Puede indicar otra carpeta con `--salida`.

En una reproducción, indique `--resultados salida_lab02_reproducida` y otra
carpeta `--salida`, después de generar un informe que corresponda a esas salidas.

El empaquetador valida ambos experimentos antes de crear el ZIP: 15 configuraciones
exitosas por dataset (estándar y dos variantes de cada método), cinco ganadores,
test reservado, predicciones coherentes con accuracy y SHA-256 de los CSV. También
regenera la preparación de Diabetes en memoria y comprueba sus tres CSV, la regla
y la procedencia; verifica que los resultados usen las 17 predictoras nativas.
Cada dataset conserva 13 archivos de evidencia. Rechaza resultados incompletos,
resultados de otra entrada, preparación alterada, fuentes LaTeX incompletas,
archivos que no son PDF y un ZIP previo con otro contenido.
Una repetición con los mismos archivos produce los mismos bytes del ZIP.

Para consultar los ejecutores y comprobar las pruebas:

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab02 --help
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 --help
.\.venv\Scripts\python.exe -m pytest -q
```

El paquete exige el archivo de versiones fijadas; los JSON de configuración
registran además las versiones de la ejecución conservada.
Consulte `docs/lab02/README.md` para repetir los experimentos en carpetas nuevas.
Ejecutarlos no actualiza el informe: sus tablas, figuras y conclusiones deben
corresponder a las salidas conservadas. `informe/main.tex` es autónomo y puede
abrirse en Overleaf; también se incluye `informe/proyecto_overleaf.zip` si existe.

El nombre del ZIP se forma con los IDs indicados mediante `--ids`.
