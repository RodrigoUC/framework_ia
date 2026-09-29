# Contraste visual de gráficos (issues #5 y #7)

## Objetivo

Corregir el contraste de gráficos y superficies en temas claro y oscuro en una rama independiente, sin alterar los resultados analíticos.

## Problema y motivo

El usuario reportó puntos, números y etiquetas indistinguibles del fondo, además de gráficos con fondo oscuro en el tema claro. Los [issues #5](https://github.com/RodrigoUC/framework_ia/issues/5) y [#7](https://github.com/RodrigoUC/framework_ia/issues/7) también exigen una fuente desplegable que se aplique una sola vez, navegación acorde al flujo, configuración visible, paneles separados por modelo y comparación de modelos realmente entrenados con hiperparámetros y precisión.

## Alcance y restricciones

- Rama: `fix/issues-5-7-visual-contrast`, desde `main` en `1386588`.
- Alcance autorizado: sistema visual local de Streamlit y gráficos Plotly, pruebas correspondientes.
- Se autorizó únicamente consultar los issues #5 y #7 con la sesión de GitHub configurada. Sin push, PR ni review de Gentle AI.
- Preservar la identidad visual existente y los cálculos de las figuras.
- TDD: no se halló configuración explícita; usar verificación funcional ordinaria con `python -m unittest discover -s tests -v`.
- Estrategia de entrega: `ask-on-risk` → `stacked-to-main`, elegida por el usuario. Cada PR integraría una unidad independiente hacia `main`; no se autorizó crearlos ni publicarlos todavía. El primer commit `18f67ec` ya excede por sí solo 400 líneas por incluir comportamiento y pruebas; una división honesta de ese work unit puede requerir `size:exception` antes de abrir su PR. No recortar pruebas ni código para ajustar el presupuesto.

## Tareas

- [x] VC-1 — Corregir de forma compartida los fondos, trazas, texto, ejes y anotaciones de Plotly para ambos temas, con pruebas de regresión. Ruta: delegada; afecta implementación y pruebas en varios archivos. Aceptación: puntos, números y letras legibles en ambos temas; ningún fondo negro heredado en modo claro; cálculos intactos. Verificación: 25 pruebas unitarias completas pasaron; 11 pruebas focalizadas repetidas por el orquestador pasaron; `git diff --check` pasó; captura HTML local en Chromium de correlación, círculo y plano ACP en ambos temas mostró fondos y etiquetas legibles. Commits: `18f67ec`, `f370118`, `3439df7`.
- [ ] VC-2 — Cerrar el sistema de contraste claro/oscuro, incluidos chips de variables y controles, sin alterar la identidad existente. Reabierta por el reporte del usuario de botones casi invisibles. Ruta: delegada; afecta tema, UI y pruebas. Aceptación: gráficos, chips, foco y botones primarios legibles en ambos temas con contraste medido, también en formularios y barra lateral. Verificación: chips >=4,5:1 en ambos temas; nuevo primario oscuro #317a57 supera 4,5:1 frente a blanco y texto configurado, y 3:1 frente a ambos fondos; 32 pruebas completas pasaron y detector Impeccable sin hallazgos. Falta comprobar estilos calculados en navegador. Commits: `0f1260f`, `e640f33`; cambio de botones pendiente de commit.
- [x] VC-3 — Unificar el flujo de fuente y configuración: fuente plegable, aplicación explícita una vez por selección, configuración de cada análisis visible cerca del trabajo, navegación ordenada y encabezado representativo en vez de «Pilares del framework». Ruta: delegada; afecta UI y pruebas. Aceptación: cambio de fuente no se aplica hasta confirmar, la configuración se encuentra sin recorrer toda la barra lateral y la navegación sigue Datos → análisis/modelos → resultados. Verificación: 4 AppTests de fuente, navegación y configuración pasaron, incluida recuperación ante CSV inicial inválido; 32 pruebas completas pasaron; revisión visual pendiente en VC-5. Commit `288883a`.
- [ ] VC-4 — Presentar modelos de clasificación en paneles independientes y comparar solo modelos entrenados/utilizados, con hiperparámetros efectivos y precisión por modelo. Ruta: delegada; afecta UI, resultados y pruebas. Aceptación: RF y Naive Bayes tienen paneles separados extensibles a más modelos; entrenar uno no borra el otro; comparación muestra nombre, parámetros usados y accuracy de cada resultado comparable sin reentrenamiento implícito. Verificación: pruebas de dominio y AppTest; commit pendiente.
- [ ] VC-5 — Verificar de extremo a extremo ambos issues y los gráficos en ambos temas; revisar el resultado visual con Impeccable y cerrar únicamente requisitos probados. Ruta: delegada para ejecución/inspección extensa. Aceptación: evidencia por punto de #5/#7, pruebas completas y sin errores en recorridos representativos. Verificación y commit pendientes.

## Progreso y evidencia

- `main` local estaba limpio y en `1386588`; se creó la rama indicada.
- Plotly ya está integrado en `main`; `_mostrar_figura` cambia fondo y fuente general, pero las factorías fijan algunos colores de trazas. Los tests de contraste previos incluyen referencias Matplotlib obsoletas.
- Se consultaron los dos issues OPEN con autorización del usuario; ambos comparten la fuente desplegable. El contraste implementado antes de leerlos solo cubre parte de su alcance.
- La implementación actual adapta trazas, ejes, anotaciones y fondos al tema; las etiquetas de mapas de calor se superponen con un color por celda sin modificar la matriz. El detector de Impeccable no reportó hallazgos. La inspección visual sigue pendiente.
- Se verificó que los gráficos circulares exponen `marker.colors`, no necesariamente `marker.color`; el adaptador contempla ambos y los nombres CSS usados por Plotly.
- Un round-trip JSON de Plotly transformó `heatmap.z` en un diccionario de arreglo tipado; el adaptador ya lo decodifica para etiquetas sin alterar los datos. Se inspeccionó una captura de seis gráficos reales (tres por tema); no sustituye una revisión de la app completa en navegador.
- El foco de botones y campos usaba verde oscuro sobre el tema oscuro (1,49:1); ahora su color se selecciona desde `st.context.theme.type`, manteniendo el borde dorado del sidebar.
- Se eligió `stacked-to-main` para futuros PRs. Límite de revisión: máximo 400 líneas por PR salvo excepción explícita. Orden previsto: (1) contraste visual; (2) flujo/fuente y navegación; (3) paneles/comparación de modelos; cada PR pendiente de autorización y de un diff aislado.
- El usuario reportó botones ilegibles después de la primera corrección: `[theme.dark]` y `[theme.dark.sidebar]` todavía usaban `primaryColor=#9effbf` con texto claro (1,2:1). El token primario oscuro ahora es #317a57; el verde brillante permanece solo como acento de foco y chips con texto oscuro. La comprobación de navegador aún no se ejecutó.

## Próximo paso

Registrar el cambio de chips y continuar con VC-3 a VC-5. Antes de abrir el primer PR, aislar su diff y resolver la posible excepción de tamaño sin omitir pruebas.
