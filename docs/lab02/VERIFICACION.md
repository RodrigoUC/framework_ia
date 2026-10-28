# Verificación Lab02

Estado al 15 de octubre de 2026 (Python 3.13.16, versiones de
`requirements.txt`).

## Cumplimiento del enunciado

| Requisito | Dónde se cumple |
| --- | --- |
| KNN, DT, RF, XGBoost y AdaBoost en el framework (clase Supervisados) | `framework_ia/modelos/supervisado/clasificacion.py`, `Clasificacion` hereda de `Supervisado` |
| Configuración estándar y variaciones | 20 configuraciones (estándar + 3 variantes por familia) y selección de atributos; informe, secciones 4 y 5 |
| Seleccionar el mejor modelo y analizarlo | Validación cruzada de 5 pliegues, ganador por familia y global; análisis por familia, discusión y conclusiones |
| Definiciones (20 %) y mapa conceptual | Informe, sección 2 y figura 1 |
| Desarrollo, experimentación y análisis (80 %) | Informe, secciones 3 a 7 y anexo |
| LaTeX / Overleaf | `informe/main.tex` y `informe/proyecto_overleaf.zip` |
| ZIP `ID-1_ID-2.zip` | `504610652_402680547_118910482_119180324.zip` |

## Comprobaciones realizadas

- `python -m pytest -q`: 142 pruebas y 59 subpruebas aprobadas.
- Repetir ambos experimentos con la semilla 42 en carpetas nuevas produjo
  `comparacion_validacion.csv` y `predicciones_test.csv` idénticos byte a byte a
  los de `salida_lab02/`.
- Sin fuga de información: las pruebas verifican que cambiar las etiquetas de
  prueba no altera la tabla de validación cruzada, que los pliegues particionan
  exactamente train y que la preparación se ajusta sólo con train.
- Cada cifra del análisis escrito se contrastó con `comparacion_validacion.csv`,
  `mejores_modelos.json` y los archivos por algoritmo.
- El informe compila con pdfLaTeX sin errores ni cajas desbordadas (31 páginas);
  se revisaron visualmente todas las páginas.
- El empaquetador valida evidencia completa, correspondencia con los CSV
  preparados (SHA-256), ganadores coherentes con la validación, exactitud
  reproducida por las predicciones, 17 predictoras en Diabetes y PDF/LaTeX
  completos; el ZIP es determinista y `MANIFIESTO.json` lista el SHA-256 de cada
  archivo.

## Resultados principales (prueba reservada)

| Dataset | Mejor por validación cruzada | F1 macro CV | F1 macro test | Exactitud test | AUC |
| --- | --- | --- | --- | --- | --- |
| Potabilidad (2011) | RF, 500 árboles | 0.644 ± 0.022 | 0.659 | 70.2 % | 0.727 |
| Diabetes (390) | RF estándar + selección de 11 atributos | 0.840 ± 0.066 | 0.764 | 88.8 % | 0.892 |

Referencias triviales (predecir siempre la clase mayoritaria): 59.6 % de
exactitud y F1 macro 0.374 en Potabilidad; 84.7 % y 0.459 en Diabetes.

## Límites conocidos

- Una sola partición de prueba; sin intervalos de confianza.
- Umbral de decisión fijo en 0.5.
- La verificación automática de Streamlit no certifica contraste visual ni uso
  con teclado.
