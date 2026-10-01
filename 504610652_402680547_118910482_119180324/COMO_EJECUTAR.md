# Cómo ejecutar

```bash
cd framework_start
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
python -m unittest discover -s tests -v
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
`salida_lab01/`. Use esos archivos para completar las tablas, figuras y el
análisis del informe LaTeX. La plantilla inicial se encuentra en
`docs/lab01/plantilla_informe.tex`; reemplácela por la plantilla oficial cuando el
profesor la facilite y complete los nombres, ID, referencias y resultados del
dataset asignado.

## Empaquetado final

Después de completar el informe PDF con la plantilla oficial y ejecutar los
experimentos, cree el ZIP con los ID reales:

```bash
python -m scripts.preparar_entrega_lab01 \
  --ids ID-1 ID-2 \
  --resultados salida_lab01 \
  --informe informe_lab01.pdf
```

El script valida que el informe sea un PDF y que existan resultados antes de
crear `ID-1_ID-2.zip`.

La aplicación acepta CSV locales ubicados en este directorio o archivos
subidos desde el navegador. Permite configurar separador, decimal, codificación
y uso de la primera columna como índice. La navegación lateral agrupa el
trabajo en Datos, Exploración y tres pilares: Agrupamiento, Clasificación y
Regresión. La regresión se muestra como próxima porque sus métodos todavía son
una plantilla.
