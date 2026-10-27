# Cómo ejecutar

```bash
cd framework_ia
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abra la dirección que Streamlit muestre en la terminal.

## Con Anaconda

Active el entorno que contiene las dependencias antes de ejecutar la
aplicación. Por ejemplo:

```bash
conda activate base
python -m streamlit run app.py
```

Para validar el framework sin abrir la interfaz:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Experimentos para Lab01

Use Python 3.11 o 3.12 y ejecute el script con el CSV asignado por el profesor:

```bash
python -m scripts.ejecutar_lab01 data/datos_profesor.csv --salida salida_lab01 --semilla 42
```

Si debe limitar las variables o cambiar el formato del CSV:

```bash
python -m scripts.ejecutar_lab01 data/datos_profesor.csv \
  --separador ";" --decimal "," \
  --features edad ingresos visitas satisfaccion \
  --salida salida_lab01 --semilla 42
```

El comando conserva parámetros, comparaciones y asignaciones en
`salida_lab01/`. Estos archivos contienen la evidencia para las tablas, figuras
y el análisis del informe LaTeX. La plantilla inicial se encuentra en
`docs/lab01/plantilla_informe.tex`.

## Empaquetado final

Con el informe PDF y los resultados experimentales, cree el ZIP con los ID
de los integrantes:

```bash
python -m scripts.preparar_entrega_lab01 \
  --ids ID-1 ID-2 \
  --resultados salida_lab01 \
  --informe informe_lab01.pdf
```

El script valida que el informe sea un PDF y que existan resultados antes de
crear `ID-1_ID-2.zip`.

## Regresión y referencia de Lab03

La referencia se incluye en scikit-learn: no hace falta descargarla ni subir un
CSV. En **Fuente de datos**, seleccione **Diabetes · referencia Lab03** y pulse
**Aplicar fuente**. Esto configura `target`, las diez predictoras, `bmi` para
regresión simple, 25 % de prueba y semilla 42; no entrena. Después revise
**Configuración de regresión** y ejecute un modelo o **Experimentos Lab03**.

Para reproducir desde la terminal:

```bash
python -m scripts.preparar_diabetes_lab03
python -m scripts.ejecutar_lab03 data/lab03/diabetes_regresion.csv --target target --feature-simple bmi --salida salida_lab03
```

La preparación guarda el CSV y `procedencia.json`; repetirla conserva bytes
idénticos y rechaza archivos distintos sin sobrescribirlos. El benchmark no
sobrescribe carpetas de resultados no vacías. Consulte
[la guía de Lab03](docs/lab03/README.md) y
[la verificación observada](docs/lab03/VERIFICACION.md).

La aplicación también acepta CSV locales ubicados en `data/` o archivos
subidos desde el navegador. Permite configurar separador, decimal, codificación
y uso de la primera columna como índice. La navegación lateral agrupa el
trabajo en Datos, Exploración y tres pilares: Agrupamiento, Clasificación y
Regresión, con configuración compartida, nueve regresores, benchmarking y
resultados guardados. Ningún cambio de fuente o navegación entrena modelos.

## Concrete · Lab03

Con el XLS oficial descargado localmente:

```powershell
python -m scripts.preparar_concrete_lab03 Concrete_Data.xls
python -m scripts.ejecutar_lab03 data/lab03/concrete/concrete.csv --target strength --feature-simple cement --salida salida_lab03/concrete --fuente https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
```

En la aplicación, cargue `data/lab03/concrete/concrete.csv` con **CSV local**.
Seleccione Regresión, objetivo `strength`, las otras ocho columnas como
predictoras y `cement` para la regresión simple. Use prueba 25 %, semilla 42
y Experimentos Lab03 con CV de cinco pliegues.

La preparación no descarga datos ni entrena.
[Reproducción del informe y resultados observados](docs/lab03/CONCRETE.md).
