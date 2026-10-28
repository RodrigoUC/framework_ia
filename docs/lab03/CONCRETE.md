# Concrete: experimentos e informe de Lab03

## Datos y protocolo

Fuente: [Concrete Compressive Strength, UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), I.-C. Yeh, DOI `10.24432/C5PK67`, licencia CC BY 4.0.

El XLS original tiene 1.030 filas, ocho predictoras numéricas, objetivo en MPa y
cero valores faltantes. Se retiran 25 duplicados de las nueve columnas, con
conservación de la primera aparición y trazabilidad de posiciones. Quedan
1.005 filas; observaciones con iguales predictoras y respuestas distintas se
conservan. [Preparación y unidades](../../data/lab03/concrete/README.md).

El protocolo utiliza 753 filas de entrenamiento y 252 de prueba, semilla 42,
sin estratificación y cinco pliegues externos mezclados. RLS utiliza `cement`;
los otros modelos utilizan las ocho predictoras. Imputación y escala se aprenden
con entrenamiento externo, sin selección de atributos. Gana el menor RMSE
promedio de validación. Las diferencias entre modelos no son pruebas de
significancia ni garantizan generalización a mezclas o laboratorios nuevos.

Las familias conceptuales son modelos lineales (RLS/RLM/Lasso/LassoCV/Ridge/
RidgeCV), vectores de soporte (SVR) y modelos basados en árboles (DT/RF).
Los nueve identificadores representan algoritmos/variantes, no nueve familias.
LassoCV/RidgeCV conservan la limitación de preprocesamiento previo a sus
pliegues internos; sólo la validación externa es evidencia comparativa.

## Entorno y reproducción

La ejecución del 21 de octubre de 2026 utilizó Python 3.11.8, numpy 2.4.6,
pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1 y xlrd 2.0.2. El entorno
completo se instala mediante `requirements.txt`.
`python -m pip check` no detectó requisitos incompatibles.

Desde la raíz, con el XLS oficial local:

```powershell
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
$env:NUMBA_NUM_THREADS="2"
python -m scripts.preparar_concrete_lab03 Concrete_Data.xls
python -m scripts.ejecutar_lab03 data/lab03/concrete/concrete.csv --target strength --feature-simple cement --salida salida_lab03/concrete --fuente https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
```

La carpeta de resultados debe estar vacía. La preparación es offline e
idempotente y conserva el XLS dentro de `data/lab03/concrete`. Se puede
repetir utilizando ese XLS y otra carpeta de resultados. La interfaz acepta el
CSV mediante su cargador existente; no necesita un preset nuevo.

Se ejecutó dos veces el benchmark completo: ambas CLI terminaron con código 0,
14 candidatos correctos y nueve configuraciones finales evaluadas. Se compararon
los 39 archivos de cada ejecución: 29 fueron idénticos byte a byte; los diez
`configuracion.json` difieren exclusivamente en `argumentos.salida`.
Al excluir sólo ese campo, coinciden todos los valores, particiones y resultados.
El registro local está en `salida_lab03/verificacion_concrete.json`.

SHA-256 del XLS original:
`710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4`.

SHA-256 del CSV canónico UTF-8/LF, sin índice:
`1eee7ee2b111a352cbadd30e49cc6a4f28e09f6ccd09636926008cefe74493a9`.

## Resultados observados

Cada fila corresponde a la configuración elegida por validación. RMSE y MAE se
expresan en MPa; R² es adimensional. DE es la desviación muestral entre pliegues.

| Modelo | Configuración | RMSE CV ± DE | RMSE prueba | MAE prueba | R² prueba |
|---|---|---:|---:|---:|---:|
| RLS | cement | 13.837660 ± 0.538692 | 15.288263 | 12.504822 | 0.193200 |
| RLM | Ocho predictoras | 10.065832 ± 0.569858 | 11.233119 | 8.937494 | 0.564438 |
| Lasso | alpha=0.1 | 10.071844 ± 0.584142 | 11.250960 | 8.964201 | 0.563053 |
| LassoCV | Cuadrícula nativa | 10.080912 ± 0.571404 | 11.235095 | 8.941900 | 0.564284 |
| Ridge | alpha=1 | 10.065763 ± 0.572251 | 11.234686 | 8.941597 | 0.564316 |
| RidgeCV | Cuadrícula nativa | 10.075443 ± 0.581873 | 11.233249 | 8.937913 | 0.564428 |
| SVR | RBF, C=10 | 7.233398 ± 0.319048 | 7.525765 | 5.382748 | 0.804498 |
| DT | Profundidad libre | 7.564658 ± 0.850092 | 6.623396 | 4.224702 | 0.848570 |
| RF | 100 árboles, profundidad libre | 5.522453 ± 0.319743 | 5.380506 | 3.650200 | 0.900070 |

Random Forest fue elegido por validación. El orden de DT y SVR cambia en prueba,
pero las decisiones permanecen determinadas por validación. Los números sin
redondear, las configuraciones y las predicciones están en los JSON/CSV locales.

## Generación del informe

```powershell
python -m scripts.generar_informe_lab03 salida_lab03/concrete --datos data/lab03/concrete/concrete.csv --procedencia data/lab03/concrete/procedencia.json --autor "Dylan Elizondo Alvarado (504610652)" --autor "Joseph Garcia Montero (402680547)" --autor "Rodrigo Ureña Castillo (118910482)" --autor "Mariana Ureña Padilla (119180324)" --fecha 2026-10-21 --salida output/lab03/informe
```

El generador no entrena. Rechaza evidencia parcial, hashes o métricas alterados,
particiones incompatibles y configuraciones ajenas al protocolo. Comprueba además
la derivación de Concrete desde el XLS, objetivo `strength`, ocho predictoras y
`cement` como predictor simple. Escribe `main.tex` autónomo y un manifiesto que
vincula el documento con los 42 archivos de evidencia/datos validados. Repetir
los mismos argumentos conserva bytes; contenido distinto no se sobrescribe.

El documento incluye los catorce candidatos, nueve evaluaciones finales, EDA,
comparación por familias, diagnósticos del ganador, límites y reproducción.
Las tablas y figuras se generan desde la evidencia validada. Fundamentos y
referencias proceden de `fundamentos.tex` y `referencias.tex`; seis de las once
referencias son artículos arbitrados (54,5 %). La fuente se compiló correctamente
con el compilador del editor integrado.


## Verificación del software

Las 82 pruebas nuevas cubren preparación: 21, integridad de evidencia: 41 e
informe: 20. La suite completa local aprobó 334 pruebas y 152 subpruebas en 249,81 s,
con permisos para los sockets locales de AppTest. El entorno restringido
bloqueaba su socketpair de asyncio antes de ejecutar la interfaz; no se
modificó el producto para eludirlo. Ruff y formato pasan en los siete archivos
Python nuevos, y pip check no encuentra requisitos incompatibles.

[CI de la implementación](https://github.com/RodrigoUC/framework_ia/actions/runs/39716522520)
aprobó 334 pruebas y 152 subpruebas en Python 3.11 y 3.12, además de compatibilidad,
ayudas CLI y reproducción offline de la referencia Diabetes. Una reproducción
adicional desde una carpeta limpia reconstruyó bytes idénticos de datos y
procedencia y resultados iguales, descontando sólo rutas de entrada/salida.
