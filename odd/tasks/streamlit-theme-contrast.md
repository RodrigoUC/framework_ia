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
- [x] **TC-2** Hacer legibles las variables seleccionadas en `st.multiselect`, fijar fondos claros para figuras Matplotlib y corregir las letras del círculo de correlación y la sobreposición ACP sin degradar las anotaciones de la matriz de calor; agregar pruebas focalizadas y verificar ambos temas. Ruta: delegada por cambios no triviales en UI y pruebas. Evidencia: 16 pruebas unitarias, AppTest de EDA, compilación Python, `git diff --check`, detector Impeccable sin hallazgos y contraste forest/papel de 11.35:1.

## Criterios de aceptación

- El modo claro conserva superficies y texto legibles; el oscuro hace legibles controles y filtros.
- Las cifras del mapa de calor mantienen contraste en celdas claras y oscuras.
- Las pruebas existentes siguen pasando.

## Verificación y entrega

- TDD: no configurado explícitamente; pruebas funcionales con `python -m unittest discover -s tests -v` y `streamlit.testing.v1.AppTest`.
- Estrategia de entrega: `ask-on-risk`; estimación inferior a 400 líneas redactadas.
- Rama: `fix/streamlit-contrast-theme`. TC-1 en commit `8f2c516` y PR #6; TC-2 en commit local `300f3eb`, todavía sin push.
- Inspección visual real de navegador: no disponible; verificación manual pendiente en ambos temas.
- Review de Gentle AI omitido por petición expresa del usuario.
- Próximo paso: revisar visualmente cuando haya navegador disponible. Push de TC-2 requiere solicitud explícita.
