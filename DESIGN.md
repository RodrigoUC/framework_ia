# Contexto de diseño · Atlas Analítico

La identidad visual vigente está en [.superdesign/design-system.md](.superdesign/design-system.md)
y los colores de Streamlit en [.streamlit/config.toml](.streamlit/config.toml).
La clasificación conserva esa identidad y controles nativos; no introduce un segundo tema.

## Usuarios y flujo principal

Estudiantes y docentes que cargan un CSV, eligen objetivo y predictoras,
comparan configuraciones y revisan evidencia reproducible. Primero se configura,
luego se ejecuta explícitamente; visitar resultados nunca entrena modelos.

## Canonical UI Map

- Datos y preparación: CSV activo, lectura explícita, exclusión de columnas,
  preparación y partición compartida.
- Exploración: EDA y ACP son destinos independientes fuera de Clustering. EDA
  conserva sus gráficos; ACP conserva su ejecución explícita y vistas.
- Clustering: K-Means, K-Medoids y HAC en destinos separados; solo algoritmos
  de agrupamiento aparecen en este grupo.
- K-Means: proyecciones opcionales t-SNE y UMAP dentro de la vista, ejecutadas
  explícitamente y coloreadas por las asignaciones activas cuando existen.
- Clasificación: una vista por KNN, Árbol de decisión, Random Forest, XGBoost,
  AdaBoost y Naive Bayes. Cada una muestra parámetros, balance del objetivo
  antes del entrenamiento, acción explícita y diagnóstico de prueba.
- Comparar configuraciones: variantes predefinidas, solo estándar o edición
  avanzada mediante JSON; revisión de candidatos antes de ejecutar y selección
  mediante validación.
- Resultados de clasificación: destino único de solo lectura para modelos
  individuales y experimentos. Separa métricas exploratorias de prueba de la
  selección por validación; muestra parámetros, procedencia, errores y descargas.
- Regresión: próxima etapa, fuera del alcance de esta implementación.

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
