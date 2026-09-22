# Cómo ejecutar

```bash
cd framework_start
python -m pip install -r requirements.txt
python -m streamlit run vca.py
```

Abra la dirección que Streamlit muestre en la terminal.

## Con Anaconda

Active el entorno que contiene las dependencias antes de ejecutar la
aplicación. Por ejemplo:

```bash
conda activate base
python -m streamlit run vca.py
```

Para validar el framework sin abrir la interfaz:

```bash
python -m unittest discover -s tests -v
```

La aplicación acepta CSV locales ubicados en este directorio o archivos
subidos desde el navegador. Permite configurar separador, decimal, codificación
y uso de la primera columna como índice. Las clases supervisadas están
declaradas como plantillas y todavía no aparecen como flujo ejecutable en la
interfaz.
