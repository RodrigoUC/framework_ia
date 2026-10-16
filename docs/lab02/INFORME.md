# Generación del informe Lab02

El informe se arma con `scripts/generar_informe_lab02.py` a partir de dos fuentes:

1. **Evidencia** de `salida_lab02/potabilidad` y `salida_lab02/diabetes`. Todas las
   tablas y figuras (comparación por familia, matrices de confusión, métricas por
   clase, ROC, importancia por permutación, efecto de la selección de atributos,
   F1 de train/CV/test, primeros niveles del árbol) se calculan desde esos
   archivos. El script valida antes que train y test sean disjuntos, que los
   pliegues cubran exactamente train y que la preparación se haya ajustado sólo
   con train.
2. **Texto redactado** en `docs/lab02/`:
   - `fundamentos.tex`: definiciones de cada método, métricas, validación cruzada,
     información mutua, desbalance, importancia por permutación y mapa conceptual.
   - `analisis/{resumen,introduccion,metodologia,potabilidad,diabetes,discusion,conclusiones,reproduccion}.tex`.
     `potabilidad.tex` y `diabetes.tex` se dividen en bloques `%% BLOQUE: clave`
     que el generador inserta junto a la tabla de cada familia.
   - `referencias.tex`: bibliografía.

```powershell
python -m scripts.generar_informe_lab02 salida_lab02 --salida output/lab02/informe `
  --autor "Dylan Elizondo Alvarado (504610652)" --autor "Joseph Garcia Montero (402680547)" `
  --autor "Rodrigo Ureña Castillo (118910482)" --autor "Mariana Ureña Padilla (119180324)"
cd output/lab02/informe
pdflatex main.tex
pdflatex main.tex
```

Salidas: `main.tex` (autónomo, sin archivos externos), `proyecto_overleaf.zip`
(para subir a Overleaf con *New Project → Upload Project*) y
`manifest_informe.json`. Paquetes LaTeX necesarios: babel (spanish), lmodern,
booktabs, tabularx, float, placeins, xurl, pgfplots, tikz y caption; todos están
en Overleaf y en TeX Live completo.

Si se vuelven a correr los experimentos, los números del texto redactado deben
revisarse contra la nueva evidencia: el generador actualiza tablas y figuras,
pero no el análisis escrito.

Si el curso exige la plantilla del profesor, copie en ella el contenido entre
`\begin{document}` y `\end{document}` de `main.tex` y el preámbulo de paquetes.
