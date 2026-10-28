# Entrega Lab02 · Clasificación supervisada

Integrantes: Dylan Elizondo Alvarado (504610652), Joseph Garcia Montero
(402680547), Rodrigo Ureña Castillo (118910482) y Mariana Ureña Padilla
(119180324).

## Contenido

| Ruta | Qué es |
| --- | --- |
| `informe/main.pdf` | Informe final (definiciones, mapa conceptual, experimentación y análisis) |
| `informe/main.tex`, `informe/proyecto_overleaf.zip` | Fuente LaTeX autónoma; el ZIP se sube tal cual a Overleaf |
| `framework_ia/` | Framework del curso; los cinco algoritmos están en `Supervisado → Clasificacion` |
| `scripts/` | Preparación de datos, experimentos, informe y empaquetado |
| `data/lab02/` | CSV originales y preparados, con `procedencia.json` (ver su `README.md`) |
| `data/ejemplo_analisis.csv` | Dataset de ejemplo que la aplicación abre al iniciar |
| `salida_lab02/` | Evidencia de los experimentos usados en el informe |
| `docs/lab02/` | Documentación, textos del informe y referencias |
| `tests/` | Pruebas automáticas del software |
| `MANIFIESTO.json` | SHA-256 y tamaño de cada archivo del ZIP |

## Abrir la aplicación

En PowerShell, dentro de la carpeta extraída (Python 3.11 o superior; los
resultados se generaron con 3.13):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Para repetir un experimento en la interfaz: cargue `data/lab02/potabilidad.csv`
o `data/lab02/diabetes.csv` (para Diabetes, pulse **Aplicar configuración
Diabetes Lab02**), abra **Experimentos Lab02**, deje la estrategia en
**Validación cruzada estratificada (5 pliegues)** y pulse **Ejecutar experimentos
Lab02**. Navegar o cambiar opciones no entrena; sólo el botón lo hace.

## Protocolo usado en el informe

1. Prueba reservada: 25 % estratificado, semilla 42. No participa en ninguna decisión.
2. Validación cruzada estratificada de 5 pliegues sobre el 75 % restante.
3. Cada familia (KNN, DT, RF, XGBoost, AdaBoost) tiene su configuración estándar y
   tres variantes que cambian un solo hiperparámetro (20 configuraciones).
4. Gana en cada familia la de mayor F1 macro promedio. Esa ganadora se vuelve a
   evaluar conservando la mitad de los atributos con mayor información mutua y se
   queda la mejor de las dos.
5. La elegida se reajusta con todo el entrenamiento y se evalúa una vez en prueba.

Imputación, one-hot, escalado y selección de atributos se ajustan sólo con los
datos de entrenamiento de cada pliegue.

## Reproducir desde cero

Las carpetas de salida deben estar vacías: los scripts no sobrescriben evidencia.

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_potabilidad_lab02 data/lab02/water_potability.csv
.\.venv\Scripts\python.exe -m scripts.preparar_diabetes_lab02 data/lab02/diabetes_original.csv
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 data/lab02/potabilidad.csv --target Potability --salida salida_reproducida/potabilidad --semilla 42 --fuente https://www.kaggle.com/datasets/adityakadiwal/water-potability
.\.venv\Scripts\python.exe -m scripts.ejecutar_lab02 data/lab02/diabetes.csv --target diabetes --features chol stab.glu hdl ratio location age gender height weight frame bp.1s bp.1d bp.2s bp.2d waist hip time.ppn --incluir-categoricas --salida salida_reproducida/diabetes --semilla 42 --fuente https://hbiostat.org/data/repo/diabetes
.\.venv\Scripts\python.exe -m pytest -q
```

Con las dependencias de `requirements.txt`, las tablas de
`salida_reproducida/` deben coincidir con las de `salida_lab02/`. Otras versiones
de las bibliotecas pueden cambiar decimales.

Para regenerar el informe y el paquete desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m scripts.generar_informe_lab02 salida_lab02 --salida output/lab02/informe --autor "Nombre Apellido (ID)"
.\.venv\Scripts\python.exe -m scripts.preparar_entrega_lab02 --ids 504610652 402680547 118910482 119180324 --informe output/lab02/informe/main.pdf --fuentes-tex output/lab02/informe/main.tex
```

`generar_informe_lab02` escribe `main.tex`; el PDF se compila con pdfLaTeX (dos
pasadas) o en Overleaf. El empaquetador rechaza la entrega si falta evidencia, si
los resultados no corresponden a los CSV preparados, si un ganador no es el mejor
en validación, si las predicciones no reproducen la exactitud registrada o si el
PDF o el LaTeX están incompletos. Repetirlo con los mismos archivos produce un ZIP
idéntico byte a byte.
