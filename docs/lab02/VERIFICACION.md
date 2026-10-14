# Verificación de Lab02 — 14 de octubre de 2026

Se verificó la implementación sobre la rama local `lab_2`, con Python 3.11
y las versiones conservadas en `requirements-lab02-lock.txt`.

- Suite completa: 110 pruebas y 52 subpruebas aprobadas. Tras ajustar la
  legibilidad del gráfico y el mensaje de resultados individuales, las
  47 pruebas de interfaz y navegación volvieron a pasar.
- Potabilidad: 3276 filas, 15 configuraciones exitosas y cinco ganadores.
  Partición efectiva: 1965 entrenamiento, 492 validación y 819 prueba.
- Diabetes: 390 filas, 15 configuraciones exitosas y cinco ganadores.
  Partición efectiva: 233 entrenamiento, 59 validación y 98 prueba.
- Semilla 42, selección por F1 macro de validación, particiones estratificadas
  y preparación aprendida únicamente con entrenamiento.
- Desde una extracción limpia del paquete, el experimento de Diabetes produjo
  los mismos 12 archivos de resultados, byte por byte. La configuración fue
  equivalente salvo la carpeta de salida. Se reutilizó el entorno de versiones
  fijadas; no se instalaron dependencias de nuevo en esa comprobación.
- La ejecución real de Potabilidad mediante Streamlit produjo los mismos
  12 archivos de resultados que la CLI, con el mismo hash del CSV y protocolo.
  El JSON de interfaz añade su información de fuente y lectura.
- Interfaz revisada en escritorio claro (1440 px), escritorio oscuro
  (1024 px) y móvil claro (390 px): sin errores JavaScript ni desbordamiento
  horizontal de la página. Las tablas amplias conservan desplazamiento interno.
- El informe procede de las exportaciones reales; su generador comprueba
  selección, particiones, matrices y métricas. La compilación y revisión visual
  del PDF se registran en `informe/verificacion_pdf.json` dentro de la entrega.
- El empaquetador comprueba datos y evidencia antes de escribir el ZIP.
  `MANIFIESTO.json` conserva el tamaño y SHA-256 de cada archivo empaquetado.

La organización del informe adapta la de Lab01 por indicación de los integrantes;
su contenido, experimentos y análisis corresponden a Lab02. La plantilla oficial
del profesor y la autorización docente para cuatro integrantes no se han
acreditado con los materiales disponibles. El proyecto está preparado para
Overleaf; no se ha cargado en Overleaf ni entregado en Moodle automáticamente.
