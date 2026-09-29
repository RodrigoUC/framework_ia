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

- [x] VC-1 — Corregir de forma compartida los fondos, trazas, texto, ejes y anotaciones de Plotly para ambos temas, con pruebas de regresión. Ruta: delegada; afecta implementación y pruebas en varios archivos. Aceptación: puntos, números y letras legibles en ambos temas; ningún fondo negro heredado en modo claro; cálculos intactos. Verificación: captura HTML local en Chromium de correlación, círculo y plano ACP en ambos temas; después, navegador real detectó razón de contraste invertida que dejaba sectores monocromáticos y etiquetas ilegibles. `9188aa3` corrigió el cociente y dio color AA al texto por sector. Chromium verificó sectores distintos y etiquetas entre 5,30:1 y 12,17:1, fondos claro/oscuro correctos tras rerun; 37 pruebas completas pasaron en ese work unit. Commits: `18f67ec`, `f370118`, `3439df7`, `9188aa3`.
- [x] VC-2 — Cerrar el sistema de contraste claro/oscuro, incluidos chips de variables y controles, sin alterar la identidad existente. Ruta: delegada; afecta tema, UI y pruebas. Aceptación: gráficos, chips, foco y botones primarios legibles en ambos temas con contraste medido, también en formularios y barra lateral. Verificación: Chromium midió botones primarios/formulario en 5,18:1 oscuro y 12,17:1 claro; chips seleccionados en 15,15:1 oscuro y 11,35:1 claro. Se quitó el CSS que anulaba colores nativos; tras cambiar tema, `Actualizar gráficos` ejecuta rerun y actualiza Plotly/CSS sin perder el dataset preparado. El botón de actualización tiene ~14:1 en ambos temas. Impeccable no reportó hallazgos. Commits: `0f1260f`, `e640f33`, `f218f3a`, `5fdd554`, `d958bf6`.
- [x] VC-3 — Unificar el flujo de fuente y configuración: fuente plegable, aplicación explícita una vez por selección, configuración de cada análisis visible cerca del trabajo, navegación ordenada y encabezado representativo en vez de «Pilares del framework». Ruta: delegada; afecta UI y pruebas. Aceptación: cambio de fuente no se aplica hasta confirmar, la configuración se encuentra sin recorrer toda la barra lateral y la navegación sigue Datos → análisis/modelos → resultados. Verificación: 4 AppTests de fuente, navegación y configuración pasaron, incluida recuperación ante CSV inicial inválido; 32 pruebas completas pasaron; revisión visual pendiente en VC-5. Commit `288883a`.
- [x] VC-4 — Presentar modelos de clasificación en paneles independientes y comparar solo modelos entrenados/utilizados, con hiperparámetros efectivos y precisión por modelo. Ruta: delegada; afecta UI y pruebas. Aceptación: RF y Naive Bayes tienen paneles separados extensibles a más modelos; entrenar uno no borra el otro; comparación muestra nombre, parámetros usados, accuracy y precisión de cada resultado comparable sin reentrenamiento implícito. Verificación: 34 pruebas completas, incluidos AppTests de entrenamiento separado, comparación sin reentrenar e invalidación de RF al cambiar parámetros; detector Impeccable sin hallazgos; `git diff --check` pasó. Commit `963c032`.
- [ ] VC-5 — Verificar de extremo a extremo ambos issues y los gráficos en ambos temas; revisar el resultado visual con Impeccable y cerrar únicamente requisitos probados. Ruta: delegada para ejecución/inspección extensa. Aceptación: evidencia por punto de #5/#7, pruebas completas y sin errores en recorridos representativos. Auditoría estática realizada; Chromium confirmó landing, gráfico circular, botones, chips y cambio de tema con rerun. La captura detectó etiquetas de métricas truncadas; `1af8ad8` las agrupó en filas 3+3+1 y Chromium confirmó las siete etiquetas completas a 1440 y 1024 px en ambos temas. Falta recorrido amplio por EDA, ACP, clasificación y agrupamiento.

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
- La clasificación presenta RF y Naive Bayes en paneles independientes. La comparación reutiliza solo resultados entrenados para el dataset y la configuración activos; muestra hiperparámetros efectivos, accuracy y precisión. Una edición de los hiperparámetros RF oculta ese resultado obsoleto sin descartar el resultado de Naive Bayes.
- La auditoría estática detectó un override CSS residual que anulaba el token oscuro en botones primarios y de navegación. Se eliminó en `5fdd554`, y la prueba nueva inspecciona el CSS inyectado además de los tokens TOML. La verificación con estilos calculados sigue pendiente.
- Chromium confirmó el contraste de botones y chips en ambos temas. También reveló que el cociente de contraste de marcas Plotly estaba invertido y forzaba sectores blancos/negros; `9188aa3` corrigió ese defecto y las etiquetas de sectores se adaptan individualmente.
- Streamlit cambia el tema del frontend sin rerun de Python; `d958bf6` añadió el botón `Actualizar gráficos` para renovar colores Plotly/CSS conservando el dataset y resultados. El recorrido en Chromium confirmó el cambio de oscuro a claro mediante ese botón.
- La tarjeta `PORCENTAJE NULOS` se truncaba en la fila de siete indicadores. `1af8ad8` la distribuyó en tres filas; Chromium confirmó todas las etiquetas a 1024 y 1440 px en ambos temas.

## Próximo paso

Completar el recorrido visual VC-5 por EDA, ACP, clasificación y agrupamiento; correr la suite final y cerrar solo requisitos comprobados. Antes de abrir el primer PR, aislar su diff y resolver la posible excepción de tamaño sin omitir pruebas.
