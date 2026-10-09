# Contexto de diseño · Atlas Analítico

La identidad visual vigente está en [.superdesign/design-system.md](.superdesign/design-system.md)
y los colores de Streamlit en [.streamlit/config.toml](.streamlit/config.toml).
LAB02 conserva esa identidad y controles nativos; no introduce un segundo tema.

## Usuarios y flujo principal

Estudiantes y docentes que cargan un CSV, eligen objetivo y predictoras,
comparan configuraciones y revisan evidencia reproducible. Primero se configura,
luego se ejecuta explícitamente; visitar resultados nunca entrena modelos.

## Canonical UI Map

- Datos y preparación: CSV activo, lectura explícita, exclusión de columnas,
  preparación y partición compartida.
- Clustering: contextual EDA + ACP, separate K-Means and K-Medoids destinations,
  and HAC. ACP runs only through its explicit action within EDA.
- K-Means visualizations: optional t-SNE and UMAP projections, each explicitly
  run and colored by the active K-Means assignments when available.
- Clasificación / RF y Naive Bayes: flujo previo conservado.
- Clasificación / Modelo individual LAB02: algoritmo, hiperparámetros visibles,
  entrenamiento explícito y diagnóstico de test.
- Clasificación / Comparar variantes LAB02: familias, estándar/variantes,
  selección por validación y ejecución explícita.
- Resultados LAB02: tabla de validación, ganadores, métricas de test, matriz de
  confusión, procedencia y descargas. Ningún entrenamiento implícito.
- Regresión: próxima etapa, fuera del alcance LAB02.

## Contratos de interacción

- La fuente cambia sólo al aplicar; cambiar datos invalida resultados derivados.
- Objetivo, features, semilla, preparación, partición o parámetros modificados
  invalidan evidencia anterior. Navegar sin cambiar configuración la conserva.
- Preparación global con imputación/escalado bloquea clasificación y muestra
  cómo restaurar el original para evitar fuga de información.
- Acciones principales identificables y errores cerca del control; estados
  vacíos explican qué falta. Errores de candidatos y de test son visibles.
- No depender únicamente del color. Las métricas macro se rotulan como macro;
  los porcentajes reflejan la partición realmente aplicada.
- Streamlit 1.65+ conserva widgets por sesión y permite descripciones accesibles
  de tablas/gráficos. Preservar estas capacidades al cambiar dependencias.

## Verificación

AppTest cubre navegación, conservación/invalidez de estado, errores, ejecución
y lectura de resultados. La revisión visual, teclado en navegador, vistas
móviles y lectores de pantalla requieren un navegador accesible; las pruebas
headless de componentes no certifican esos aspectos.
