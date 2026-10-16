# Contexto de diseño · Nexo IA

La identidad visual vigente está en [.superdesign/design-system.md](.superdesign/design-system.md)
y los colores de Streamlit en [.streamlit/config.toml](.streamlit/config.toml).
Clasificación y regresión conservan esa identidad y controles nativos; no introducen otro tema.

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
  AdaBoost y Naive Bayes. Cada una muestra parámetros, acción explícita y
  diagnóstico de prueba; objetivo, balance y preparación pertenecen a la
  configuración compartida.
- Experimentos Lab02: reutiliza esa configuración para comparar 15 candidatos
  (estándar y dos variantes de cinco familias) con una acción explícita. Reutiliza
  validación externa o la extrae sólo de train. Tabla y gráfica de validación
  preceden a métricas de prueba, matrices y predicciones de los ganadores;
  descargas ZIP/CSV/JSON usan el mismo formato que la CLI.
- Comparar modelos entrenados: destino de solo lectura que muestra sólo modelos
  ya entrenados. Random Forest conserva snapshots independientes de `gini` y
  `entropy`; no ejecuta nuevos ajustes.
- Resultados de clasificación: destino de solo lectura para ejecuciones
  individuales y experimentos guardados, en apartados separados. Muestra modelo,
  parámetros, métricas, partición y procedencia; nunca invoca entrenamiento.
- Configuración de regresión: target numérico, predictores, predictor simple,
  semilla, partición y preparación aprendida solo con train.
- Regresión: RLS, RLM, Lasso, LassoCV, Ridge, RidgeCV, SVR, DT y RF en destinos
  separados. Entrenamiento explícito, métricas numéricas y diagnósticos de prueba.
- Experimentos Lab03: catorce candidatos, selección por RMSE de validación,
  comparación con dispersión por pliegue y prueba final de ganadores; descargas
  ZIP/CSV/JSON usan los exportadores compartidos con CLI.
- Comparar regresores entrenados / Resultados de regresión: solo lectura,
  sin ajustar modelos; resultados incluye también el experimento guardado.

## Contratos de interacción

- Flujo guiado: **Fuente → Configuración → Ejecución → Resultados**. Cada vista
  supervisada muestra etapa, objetivo/predictores/split efectivos y siguiente
  acción. La disponibilidad se deriva de datos y configuración, sin flags de
  pantallas visitadas ni confirmaciones adicionales.
- Los destinos dependientes y sus acciones se deshabilitan ante requisitos
  incumplidos, con razones visibles y acceso permanente a Datos/Configuración
  para corregirlos. Las rutas directas verifican los mismos requisitos.
- Preparación global y partición manual son opcionales. EDA, ACP y clustering
  son ramas independientes, no pasos obligatorios antes de los laboratorios.
  Clasificación requiere clases discretas; regresión un objetivo numérico real
  válido. No se convierten objetivos automáticamente.
- Los diagnósticos de disponibilidad no ajustan modelos ni transformadores.
  Un candidato inviable no impide ejecutar otros candidatos preparados de un
  benchmark. Tras ejecutar, la navegación de resultados se actualiza sin otro ajuste.

- La fuente cambia sólo al aplicar; cambiar datos invalida resultados derivados.
- Diabetes de referencia Lab03 se carga localmente, sin red ni entrenamiento.
  Aplicar consume un preset de regresión una sola vez después de sincronizar
  los datos; navegar no vuelve a imponerlo sobre los ajustes manuales.
- Objetivo, features, semilla, preparación, partición o fuente modificados
  invalidan evidencia incomparable. Los cambios de parámetros de widgets solos
  no sustituyen la última ejecución guardada; sólo un entrenamiento explícito
  la reemplaza. Navegar sin cambiar contexto la conserva.
- Las opciones experimentales modificadas se señalan sin alterar la evidencia
  ejecutada. Un intento fallido conserva el último experimento exitoso del mismo
  contexto; los fallos de candidatos se muestran como ejecución parcial.
- En regresión, cambiar el predictor simple también invalida la evidencia.
  Si falla la prueba del ganador seleccionado por validación, se muestra el
  intento parcial y se conserva la ejecución completa anterior; nunca se
  sustituye al ganador por otro elegido usando test.
- Preparación global con imputación/escalado bloquea clasificación y regresión y muestra
  cómo restaurar el original para evitar fuga de información.
- Acciones principales identificables y errores cerca del control; estados
  vacíos explican qué falta. Errores de candidatos y de test son visibles.
- No depender únicamente del color. Las métricas macro se rotulan como macro;
  accuracy y recall por clase reflejan la partición realmente aplicada. Las
  etiquetas `y`/`n` sólo se asignan a clases de texto `y`/`n`; otras clases
  conservan sus etiquetas literales y clases sin soporte se marcan no evaluadas.
- Streamlit 1.65+ conserva widgets por sesión y permite descripciones accesibles
  de tablas/gráficos. Preservar estas capacidades al cambiar dependencias.
- RMSE, MAE, MSE y R² se muestran como números, no porcentajes. Los residuos
  son real menos predicción. Los controles de ambas familias están aislados.

## Verificación

AppTest cubre navegación, conservación/invalidez de estado, errores, ejecución
y lectura de resultados. La revisión visual, teclado en navegador, vistas
móviles y lectores de pantalla requieren un navegador accesible; las pruebas
headless de componentes no certifican esos aspectos.
