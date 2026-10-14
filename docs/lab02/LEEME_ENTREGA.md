# Entrega Lab02

El ZIP contiene los CSV originales y su procedencia, el framework, la aplicación,
los ejecutores, pruebas de software, resultados de Potabilidad y Diabetes, y el
informe PDF junto con su fuente LaTeX. `MANIFIESTO.json` permite comprobar el
SHA-256 y tamaño de cada archivo; no incorpora rutas absolutas ni fechas variables.

Para abrir la aplicación tras extraer el ZIP, abra PowerShell en la carpeta extraída:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lab02-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La configuración de tema Atlas se conserva en `.streamlit/config.toml`.

Para preparar el paquete desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_entrega_lab02 --ids 504610652 402680547 118910482 119180324 --informe output/lab02/informe/main.pdf --fuentes-tex output/lab02/informe/main.tex
```

En una copia extraída del ZIP, use `--informe informe/main.pdf` y
`--fuentes-tex informe/main.tex`. Puede indicar otra carpeta con `--salida`.

El ejecutor valida ambos experimentos antes de crear el ZIP: 15 configuraciones
exitosas por dataset (estándar y dos variantes de cada método), cinco ganadores,
test reservado, predicciones coherentes con accuracy y SHA-256 de los CSV.
Cada dataset conserva 13 archivos de evidencia. Rechaza resultados incompletos,
fuentes LaTeX incompletas, archivos que no son PDF y un ZIP previo con otro contenido.
Una repetición con los mismos archivos produce los mismos bytes del ZIP.

Después de extraer el ZIP, abra PowerShell en la carpeta extraída:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lab02-lock.txt
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 --help
.\.venv\Scripts\python.exe -m pytest -q
```

El paquete exige el archivo de versiones fijadas; los JSON de configuración
registran además las versiones de la ejecución conservada.
Consulte `docs/lab02/README.md` para repetir los experimentos en carpetas nuevas.
Ejecutarlos no actualiza el informe: sus tablas, figuras y conclusiones deben
corresponder a las salidas conservadas. `informe/main.tex` es autónomo y puede
abrirse en Overleaf; también se incluye `informe/proyecto_overleaf.zip` si existe.

El paquete usa los cuatro IDs indicados por los integrantes. El enunciado permite
trabajo individual o de hasta dos personas; la autorización docente para cuatro
integrantes permanece por confirmar.
