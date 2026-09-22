# Sistema de diseño — Framework IA

## Producto y usuarios

Aplicación de escritorio para estudiantes y docentes que cargan un CSV,
preparan datos y ejecutan análisis exploratorio, clustering, reducción de
dimensionalidad o clasificación. El trabajo principal es comparar resultados,
entender calidad de datos y exportar evidencia reproducible para Lab01.

## Problemas UX a resolver

1. La configuración global y la específica se mezclan en la barra lateral.
2. Nueve pestañas no comunican el orden recomendado ni el estado de avance.
3. Métricas, gráficos y tablas no tienen una jerarquía consistente para
   interpretar o comparar resultados.
4. Los resultados persisten en sesión pero no se percibe con claridad qué
   configuración los produjo ni cuándo quedaron desactualizados.

## Dirección de experiencia

Nueva identidad: **Atlas Analítico**. Un lenguaje técnico-minimalista inspirado
en planos de ingeniería y cuadernos de laboratorio, con jerarquía editorial y
una cuadrícula bento sutil. Convertir el flujo en una secuencia visible:
**1. Datos → 2. Preparar → 3. Explorar → 4. Modelar → 5. Comparar y exportar**.
Usar una navegación lateral por grupos, un resumen fijo del dataset y una zona
de resultados con: indicadores primero, visualización principal después y
detalle tabular/descarga al final. Los controles avanzados deben permanecer
colapsados hasta que el usuario los necesite.

## Tokens visuales obligatorios

- Encabezados: `Space Grotesk`; texto: `General Sans` o sans-serif; etiquetas,
  parámetros, semillas y metadatos: `JetBrains Mono`.
- Primario bosque: `#1A3C2B`; papel: `#F7F7F5`; tinta: `#242523`; cuadrícula:
  `#3A3A38` al 20% de opacidad.
- Acentos semánticos: menta `#9EFFBF` (resultado listo), coral `#FF8C69`
  (alerta/variación) y oro `#F4D35E` (selección o recomendación).
- Tema oscuro: fondo `#111713`, superficie `#1A241D`, texto `#F7F7F5`, borde
  `#B6C4B9` al 25%, con el mismo verde y acentos para consistencia.
- Radio 0–2 px. Sin sombras ni gradientes; usar bloques planos, bordes de 1 px
  y amplio espacio negativo.
- La cuadrícula de fondo solo debe ser sutil y nunca comprometer legibilidad de
  tablas o gráficos.

## Patrones

- Encabezado compacto con marca geométrica, nombre del dataset, filas,
  columnas, nulos y estado de preparación.
- Navegación multinivel y expandible: Datos; Exploración; y los tres pilares
  del framework: **Agrupamiento**, **Clasificación** y **Regresión**. El segundo
  nivel presenta familias y técnicas disponibles:
  - Agrupamiento → Particional → K-Means, K-Medoids; Jerárquico → HAC.
  - Exploración → Reducción dimensional → ACP, t-SNE, UMAP.
  - Clasificación → Random Forest, Naive Bayes.
  - Regresión → lineal simple y múltiple (marcadas como próximas mientras sean
    plantillas).
  El ítem activo debe mostrar toda la ruta (breadcrumb), y el menú conserva
  sus grupos abiertos al navegar entre técnicas hermanas. Etiquetas técnicas
  en mayúscula con índice.
- Acción principal única por vista: “Ejecutar análisis”, en verde bosque.
- Cada resultado muestra configuración/semilla junto a las métricas para hacer
  la comparación trazable, con una insignia de estado monoespaciada.
- Gráficos grandes en paneles bento, con una lectura o conclusión breve debajo;
  tablas en expandibles y CSV como acción secundaria.
- En pantallas estrechas, apilar controles antes de resultados y preservar la
  acción principal visible.
