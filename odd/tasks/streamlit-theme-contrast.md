# Contraste de temas en Streamlit

## Objetivo

Recuperar la legibilidad de filtros, resultados y matriz de correlación en los temas claro y oscuro sin cambiar la identidad visual existente.

## Problema y causa observada

El CSS fija el sidebar oscuro y fuerza colores sobre todos sus descendientes y controles. El postprocesado nocturno de Matplotlib pinta de blanco todas las anotaciones del mapa de calor, incluso las que aparecen sobre celdas claras.

## Alcance autorizado

- Ajustar `.streamlit/config.toml` y la capa visual de `framework_ia/ui/streamlit_app.py` para respetar el tema activo.
- Ajustar las anotaciones de la matriz de correlación sin alterar su cálculo.
- Añadir pruebas focalizadas de tema y contraste.
- Preservar los cambios preexistentes en `.agents/`, `.claude/` y `.atl/`.

## Tarea

- [x] **TC-1** Corregir contraste en ambos temas para filtros y mapa de calor; verificar con pruebas, AppTest y auditoría de diseño. Ruta: delegada por edición no trivial en varios archivos. Evidencia: 13 pruebas unitarias, AppTest de EDA, compilación Python, `git diff --check`, detector Impeccable sin hallazgos y contraste calculado de 12.03:1 o superior para texto sobre superficies principales y laterales.

## Criterios de aceptación

- El modo claro conserva superficies y texto legibles; el oscuro hace legibles controles y filtros.
- Las cifras del mapa de calor mantienen contraste en celdas claras y oscuras.
- Las pruebas existentes siguen pasando.

## Verificación y entrega

- TDD: no configurado explícitamente; pruebas funcionales con `python -m unittest discover -s tests -v` y `streamlit.testing.v1.AppTest`.
- Estrategia de entrega: `ask-on-risk`; estimación inferior a 400 líneas redactadas.
- Rama: `fix/streamlit-contrast-theme`. Commit de unidad y revisión: pendientes.
- Inspección visual real de navegador: no disponible; verificación manual pendiente en ambos temas.
- Próximo paso: registrar el commit local y revisar la interfaz en navegador cuando esté disponible.
