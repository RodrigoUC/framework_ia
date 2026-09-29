# Contraste visual de gráficos (issues #5 y #7)

## Objetivo

Corregir el contraste de gráficos y superficies en temas claro y oscuro en una rama independiente, sin alterar los resultados analíticos.

## Problema y motivo

El usuario reportó puntos, números y etiquetas indistinguibles del fondo, además de gráficos con fondo oscuro en el tema claro. El contenido exacto de los issues #5 y #7 queda pendiente de consulta autorizada.

## Alcance y restricciones

- Rama: `fix/issues-5-7-visual-contrast`, desde `main` en `1386588`.
- Alcance autorizado: sistema visual local de Streamlit y gráficos Plotly, pruebas correspondientes.
- Sin acceso remoto, push, PR ni review de Gentle AI.
- Preservar la identidad visual existente y los cálculos de las figuras.
- TDD: no se halló configuración explícita; usar verificación funcional ordinaria con `python -m unittest discover -s tests -v`.
- Estrategia de entrega: `ask-on-risk`; previsión aproximada: menos de 400 líneas, reevaluar con el diff real.

## Tareas

- [ ] VC-1 — Corregir de forma compartida los fondos, trazas, texto, ejes y anotaciones de Plotly para ambos temas, con pruebas de regresión. Ruta: delegada; afecta implementación y pruebas en varios archivos. Aceptación: puntos, números y letras legibles en ambos temas; ningún fondo negro heredado en modo claro; cálculos intactos. Verificación: 22 pruebas unitarias completas pasaron; 8 pruebas focalizadas repetidas por el orquestador pasaron; `git diff --check` pasó; comprobación visual representativa pendiente porque la exportación PNG de Plotly falló en este entorno. Commit: pendiente.
- [ ] VC-2 — Auditar la coherencia final de los colores de Streamlit y figuras, cubrir rutas de gráficos faltantes y contrastar el resultado con el alcance exacto de los issues cuando esté autorizado. Ruta: delegada si requiere varias ediciones no triviales. Aceptación: temas claro/oscuro consistentes en las vistas afectadas y evidencia explícita para #5 y #7. Verificación: pruebas completas, detector Impeccable y revisión visual acotada. Commit: pendiente.

## Progreso y evidencia

- `main` local estaba limpio y en `1386588`; se creó la rama indicada.
- Plotly ya está integrado en `main`; `_mostrar_figura` cambia fondo y fuente general, pero las factorías fijan algunos colores de trazas. Los tests de contraste previos incluyen referencias Matplotlib obsoletas.
- Se pidió autorización para leer los issues con la sesión de GitHub configurada; aún no hubo respuesta.
- La implementación actual adapta trazas, ejes, anotaciones y fondos al tema; las etiquetas de mapas de calor se superponen con un color por celda sin modificar la matriz. El detector de Impeccable no reportó hallazgos. La inspección visual sigue pendiente.

## Próximo paso

Implementar VC-1 localmente y verificar sus resultados sin suponer el texto de los issues.
