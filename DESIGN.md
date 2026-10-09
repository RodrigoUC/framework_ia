# Contexto de diseño · Atlas Analítico

La identidad visual vigente está en [.superdesign/design-system.md](.superdesign/design-system.md)
y los colores de Streamlit en [.streamlit/config.toml](.streamlit/config.toml).
La clasificación conserva esa identidad y controles nativos; no introduce un segundo tema.

## Usuarios y flujo principal

Estudiantes y docentes que cargan un CSV, eligen objetivo y predictoras,
entrenan explícitamente y revisan evidencia reproducible. Los destinos de
comparación/resultados leen ejecuciones guardadas; visitarlos nunca entrena.

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
- Comparar modelos entrenados: destino de solo lectura que muestra sólo modelos
  ya entrenados. Random Forest conserva snapshots independientes de `gini` y
  `entropy`; no ejecuta nuevos ajustes.
- Resultados de clasificación: destino de solo lectura para ejecuciones
  individuales guardadas, con accuracy general y recall porcentual por clase
  real. Muestra modelo, parámetros, partición y procedencia. La API offline
  `Clasificacion.experimentar(...)` conserva la selección por validación y no
  se invoca desde estas rutas.
- Regresión: próxima etapa, fuera del alcance de esta implementación.

## Contratos de interacción

- La fuente cambia sólo al aplicar; cambiar datos invalida resultados derivados.
- Objetivo, features, semilla, preparación, partición o fuente modificados
  invalidan evidencia incomparable. Los cambios de parámetros de widgets solos
  no sustituyen la última ejecución guardada; sólo un entrenamiento explícito
  la reemplaza. Navegar sin cambiar contexto la conserva.
- Preparación global con imputación/escalado bloquea clasificación y muestra
  cómo restaurar el original para evitar fuga de información.
- Acciones principales identificables y errores cerca del control; estados
  vacíos explican qué falta. Errores de candidatos y de test son visibles.
- No depender únicamente del color. Las métricas macro se rotulan como macro;
  accuracy y recall por clase reflejan la partición realmente aplicada. Las
  etiquetas `y`/`n` sólo se asignan a clases de texto `y`/`n`; otras clases
  conservan sus etiquetas literales y clases sin soporte se marcan no evaluadas.
- Streamlit 1.65+ conserva widgets por sesión y permite descripciones accesibles
  de tablas/gráficos. Preservar estas capacidades al cambiar dependencias.

## Verificación

AppTest cubre navegación, conservación/invalidez de estado, errores, ejecución
y lectura de resultados. La revisión visual, teclado en navegador, vistas
móviles y lectores de pantalla requieren un navegador accesible; las pruebas
headless de componentes no certifican esos aspectos.
