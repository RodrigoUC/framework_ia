"""Construye el informe LaTeX de LAB 02 a partir de la evidencia exportada.

No entrena, no modifica datasets y no descarga nada. Antes de escribir
comprueba que la evidencia sea coherente (particiones disjuntas, selección por
validación cruzada, matrices y métricas consistentes). Las tablas y figuras se
generan con los números exportados; el análisis interpretativo se toma de los
archivos redactados en ``docs/lab02/analisis``. El resultado compila con
pdfLaTeX en Overleaf sin imágenes ni archivos externos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALGORITMOS = ("KNN", "DT", "RF", "XGBoost", "AdaBoost")
NOMBRES = {
    "KNN": "K vecinos más cercanos (KNN)",
    "DT": "Árbol de decisión (DT)",
    "RF": "Random Forest (RF)",
    "XGBoost": "XGBoost",
    "AdaBoost": "AdaBoost",
}
TITULOS = {"potabilidad": "Potabilidad del agua", "diabetes": "Diabetes"}
DESCRIPCION_VARIANTES = {
    "estandar": "valores predeterminados de la biblioteca",
    "k_15": "$k=15$ vecinos (estándar: 5)",
    "pesos_distancia": "votos ponderados por $1/d$ (estándar: uniformes)",
    "manhattan": "distancia Manhattan, $p=1$ (estándar: euclidiana)",
    "profundidad_5": "profundidad máxima 5 (estándar: sin límite)",
    "hoja_minima_10": "mínimo 10 observaciones por hoja (estándar: 1)",
    "clases_balanceadas": "pesos por observación inversos a la frecuencia de su clase",
    "500_arboles": "500 árboles (estándar: 100)",
    "hoja_minima_5": "mínimo 5 observaciones por hoja (estándar: 1)",
    "tasa_0_05": "tasa de aprendizaje 0.05 (estándar: 0.3)",
    "profundidad_3": "profundidad máxima 3 (estándar: 6)",
    "200_estimadores": "200 estimadores (estándar: 50)",
    "tasa_0_5": "tasa de aprendizaje 0.5 (estándar: 1.0)",
}
COLORES = {"KNN": "blue!70!black", "DT": "orange!85!black", "RF": "green!55!black",
           "XGBoost": "red!75!black", "AdaBoost": "violet!80!black"}
RAIZ = Path(__file__).resolve().parents[1]
RECURSOS = RAIZ / "docs" / "lab02"


@dataclass
class Evidencia:
    nombre: str
    ruta: Path
    configuracion: dict[str, Any]
    comparacion: list[dict[str, Any]]
    mejores: dict[str, Any]
    matrices: dict[str, tuple[list[str], list[list[int]]]]
    roc: dict[str, list[tuple[float, float]]]
    permutacion: dict[str, list[dict[str, Any]]]
    hashes: dict[str, str]

    @property
    def experimento(self) -> dict[str, Any]:
        return self.configuracion["experimento"]


# ---------------------------------------------------------------- utilidades


def tex(valor: Any) -> str:
    """Escapa texto de JSON/CSV, sin interpretarlo como código LaTeX."""
    mapa = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapa.get(c, c) for c in str(valor)).replace("\n", " ")


def numero(valor: Any) -> float:
    resultado = float(valor)
    if not math.isfinite(resultado):
        raise ValueError(f"Valor no finito: {valor!r}")
    return resultado


def p4(valor: Any) -> str:
    return f"{numero(valor):.4f}"


def pct(valor: Any) -> str:
    return f"{100 * numero(valor):.1f}" + r"\%"


def verdadero(valor: Any) -> bool:
    return str(valor).lower() in ("true", "1")


def _json(ruta: Path) -> dict[str, Any]:
    def rechazar(valor: str) -> None:
        raise ValueError(f"JSON no estándar en {ruta.name}: {valor}")

    contenido = json.loads(ruta.read_text(encoding="utf-8-sig"), parse_constant=rechazar)
    if not isinstance(contenido, dict):
        raise TypeError(f"Se esperaba un objeto JSON: {ruta}")
    return contenido


def _hash(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _csv(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(encoding="utf-8-sig", newline="") as entrada:
        return list(csv.DictReader(entrada))


def _huella(huella: str) -> str:
    return r"{\footnotesize\texttt{" + huella[:32] + r"}\newline\texttt{" + huella[32:] + "}}"


# --------------------------------------------------------------- validación


def cargar_evidencia(nombre: str, ruta: Path) -> Evidencia:
    configuracion = _json(ruta / "configuracion.json")
    mejores = _json(ruta / "mejores_modelos.json")
    comparacion = _csv(ruta / "comparacion_validacion.csv")
    exp = configuracion["experimento"]
    if exp.get("estrategia_seleccion") != "validacion_cruzada_estratificada":
        raise ValueError(f"{nombre}: el informe requiere la evidencia de validación cruzada.")
    if exp.get("test_usado_para_seleccion") is not False:
        raise ValueError(f"{nombre}: test no puede participar en la selección.")
    if exp.get("preprocesamiento", {}).get("ajustado_solo_train") is not True:
        raise ValueError(f"{nombre}: no se acredita preparación ajustada sólo con train.")
    train, test = set(map(str, exp["train_indices"])), set(map(str, exp["test_indices"]))
    if not train or not test or train & test:
        raise ValueError(f"{nombre}: train y test deben ser no vacíos y disjuntos.")
    pliegues = [set(map(str, p)) for p in exp["pliegues_indices"]]
    if set().union(*pliegues) != train or sum(map(len, pliegues)) != len(train):
        raise ValueError(f"{nombre}: los pliegues no particionan exactamente train.")
    if set(mejores) != set(ALGORITMOS):
        raise ValueError(f"{nombre}: faltan ganadores de las cinco familias.")
    matrices, roc, permutacion = {}, {}, {}
    hashes = {a: _hash(ruta / a) for a in ("configuracion.json", "comparacion_validacion.csv",
                                            "mejores_modelos.json")}
    for algoritmo in ALGORITMOS:
        filas = [f for f in comparacion if f["algoritmo"] == algoritmo]
        elegidas = [f for f in filas if verdadero(f["seleccionado"])]
        if len(elegidas) != 1 or any(f["estado"] != "ok" for f in filas):
            raise ValueError(f"{nombre}/{algoritmo}: selección ambigua o candidatos fallidos.")
        maximo = max(numero(f["puntuacion_validacion"]) for f in filas)
        if not math.isclose(numero(elegidas[0]["puntuacion_validacion"]), maximo, abs_tol=1e-12):
            raise ValueError(f"{nombre}/{algoritmo}: el ganador no maximiza la validación cruzada.")
        mejor = mejores[algoritmo]
        if elegidas[0]["configuracion"].split("+")[0] != mejor["metadatos"]["configuracion"]:
            raise ValueError(f"{nombre}/{algoritmo}: el modelo de test no es el ganador.")
        contenido = list(csv.reader((ruta / algoritmo / "matriz_confusion_test.csv").open(encoding="utf-8")))
        etiquetas = contenido[0][1:]
        matriz = [[int(c) for c in fila[1:]] for fila in contenido[1:]]
        total = sum(map(sum, matriz))
        if total != len(test) or total != mejor["muestra_test"]:
            raise ValueError(f"{nombre}/{algoritmo}: la matriz no cubre test.")
        accuracy = sum(matriz[i][i] for i in range(len(matriz))) / total
        if not math.isclose(accuracy, mejor["metricas_test"]["accuracy"], abs_tol=1e-10):
            raise ValueError(f"{nombre}/{algoritmo}: accuracy no coincide con la matriz.")
        f1s = []
        for i, etiqueta in enumerate(etiquetas):
            soporte, predichas, tp = sum(matriz[i]), sum(f[i] for f in matriz), matriz[i][i]
            f1 = 2 * tp / (soporte + predichas) if soporte + predichas else 0.0
            registrado = mejor["metricas_test"]["precision"]["por_clase"][etiqueta]["f1"]
            if not math.isclose(f1, registrado, abs_tol=1e-10):
                raise ValueError(f"{nombre}/{algoritmo}: F1 de {etiqueta} no coincide con la matriz.")
            f1s.append(f1)
        if not math.isclose(sum(f1s) / len(f1s), mejor["metricas_test"]["f1"]["macro"], abs_tol=1e-10):
            raise ValueError(f"{nombre}/{algoritmo}: F1 macro no coincide con la matriz.")
        matrices[algoritmo] = (etiquetas, matriz)
        roc[algoritmo] = [(numero(f["fpr"]), numero(f["tpr"])) for f in _csv(ruta / algoritmo / "roc_test.csv")]
        permutacion[algoritmo] = _csv(ruta / algoritmo / "importancia_permutacion.csv")
        for archivo in ("matriz_confusion_test.csv", "predicciones_test.csv", "roc_test.csv",
                        "importancia_permutacion.csv"):
            hashes[f"{algoritmo}/{archivo}"] = _hash(ruta / algoritmo / archivo)
    return Evidencia(nombre, ruta, configuracion, comparacion, mejores, matrices, roc, permutacion, hashes)


# ------------------------------------------------------------- bloques LaTeX


def tabla(encabezados, filas, *, caption, alineacion=None, etiqueta=None, tamano=r"\small"):
    alineacion = alineacion or ("l" + "r" * (len(encabezados) - 1))
    lineas = [r"\begin{table}[H]\centering" + tamano, r"\caption{" + caption + "}"]
    if etiqueta:
        lineas.append(r"\label{" + etiqueta + "}")
    lineas += [r"\begin{tabular}{" + alineacion + r"}\toprule", " & ".join(encabezados) + r" \\ \midrule"]
    lineas += [" & ".join(fila) + r" \\" for fila in filas]
    lineas += [r"\bottomrule\end{tabular}", r"\end{table}"]
    return "\n".join(lineas)


def _tabular(encabezados, filas, alineacion) -> str:
    lineas = [r"\begin{tabular}{" + alineacion + r"}\toprule", " & ".join(encabezados) + r" \\ \midrule"]
    lineas += [" & ".join(fila) + r" \\" for fila in filas]
    return "\n".join([*lineas, r"\bottomrule\end{tabular}"])


def bloques_analisis(nombre: str) -> dict[str, str]:
    """Separa un archivo de análisis redactado en bloques ``%% BLOQUE: clave``."""
    ruta = RECURSOS / "analisis" / f"{nombre}.tex"
    bloques: dict[str, list[str]] = {}
    actual = None
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        coincidencia = re.match(r"^%% BLOQUE: (\S+)", linea)
        if coincidencia:
            actual = coincidencia.group(1)
            bloques[actual] = []
        elif actual is not None:
            bloques[actual].append(linea)
    return {clave: "\n".join(lineas).strip() for clave, lineas in bloques.items()}


def _filas_familia(evidencia: Evidencia, algoritmo: str) -> list[dict[str, Any]]:
    return [f for f in evidencia.comparacion if f["algoritmo"] == algoritmo]


def _elegida(evidencia: Evidencia, algoritmo: str) -> dict[str, Any]:
    return next(f for f in _filas_familia(evidencia, algoritmo) if verdadero(f["seleccionado"]))


def _base_estandar(evidencia: Evidencia, algoritmo: str) -> dict[str, Any]:
    return next(f for f in _filas_familia(evidencia, algoritmo)
                if f["etapa"] == "configuracion" and f["configuracion"] == "estandar")


def _ganadora_configuracion(evidencia: Evidencia, algoritmo: str) -> dict[str, Any]:
    filas = [f for f in _filas_familia(evidencia, algoritmo) if f["etapa"] == "configuracion"]
    return max(filas, key=lambda f: numero(f["puntuacion_validacion"]))


def _referencia_mayoritaria(evidencia: Evidencia) -> tuple[str, float, float]:
    distribucion = evidencia.experimento["distribuciones"]
    mayoritaria = max(distribucion["train"], key=distribucion["train"].get)
    proporcion = distribucion["test"][mayoritaria] / evidencia.experimento["muestra_test"]
    clases = len(distribucion["train"])
    return mayoritaria, proporcion, (2 * proporcion / (1 + proporcion)) / clases


def _nombre_clase(evidencia: Evidencia, clase: str) -> str:
    if evidencia.nombre == "potabilidad":
        return {"0": "No potable (0)", "1": "Potable (1)"}.get(str(clase), str(clase))
    return str(clase)


def figura_cv_test(evidencia: Evidencia) -> str:
    cv, prueba, entrenamiento = [], [], []
    for algoritmo in ALGORITMOS:
        elegida = _elegida(evidencia, algoritmo)
        mejor = evidencia.mejores[algoritmo]
        cv.append(f"({algoritmo},{numero(elegida['puntuacion_validacion']):.4f}) +- (0,{numero(elegida['puntuacion_validacion_std']):.4f})")
        prueba.append(f"({algoritmo},{mejor['metricas_test']['f1']['macro']:.4f})")
        entrenamiento.append(f"({algoritmo},{mejor['metadatos']['analisis']['metricas_train']['f1_macro']:.4f})")
    _, _, f1_base = _referencia_mayoritaria(evidencia)
    return r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[ybar, bar width=8pt, width=\linewidth, height=6.2cm, ymin=0, ymax=1.08,
 ylabel={F1 macro}, symbolic x coords={KNN,DT,RF,XGBoost,AdaBoost}, xtick=data,
 enlarge x limits=0.14, ymajorgrids, legend style={at={(0.5,-0.16)},anchor=north,legend columns=4,font=\small},
 tick label style={font=\small}]
\addplot[fill=gray!35, draw=gray!70!black] coordinates {""" + " ".join(entrenamiento) + r"""};
\addplot[fill=blue!55, draw=blue!70!black, error bars/.cd, y dir=both, y explicit] coordinates {""" + " ".join(cv) + r"""};
\addplot[fill=orange!70, draw=orange!80!black] coordinates {""" + " ".join(prueba) + r"""};
\draw[dashed, red!70!black, thick] ({rel axis cs:0,0}|-{axis cs:KNN,""" + f"{f1_base:.4f}" + r"""}) -- ({rel axis cs:1,0}|-{axis cs:KNN,""" + f"{f1_base:.4f}" + r"""});
\legend{Entrenamiento, Validación cruzada (media $\pm$ d.e.), Test}
\end{axis}\end{tikzpicture}
\caption{""" + TITULOS[evidencia.nombre] + r""": F1 macro del modelo elegido en cada familia. La línea roja es la referencia que siempre predice la clase mayoritaria (F1 macro """ + f"{f1_base:.3f}" + r"""). La distancia entre la barra gris y la azul indica sobreajuste.}
\label{fig:resumen-""" + evidencia.nombre + r"""}
\end{figure}"""


def figura_roc(evidencia: Evidencia) -> str:
    curvas = []
    for algoritmo in ALGORITMOS:
        puntos = " ".join(f"({x:.4f},{y:.4f})" for x, y in evidencia.roc[algoritmo])
        auc = evidencia.mejores[algoritmo]["metadatos"]["analisis"]["roc"]["auc"]
        curvas.append(r"\addplot[thick, " + COLORES[algoritmo] + "] coordinates {" + puntos + "};")
        curvas.append(r"\addlegendentry{" + algoritmo + f" (AUC {auc:.3f})" + "}")
    positiva = evidencia.mejores["RF"]["metadatos"]["analisis"]["roc"]["clase_positiva"]
    return r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[width=0.72\linewidth, height=0.62\linewidth, xmin=0, xmax=1, ymin=0, ymax=1,
 xlabel={Tasa de falsos positivos}, ylabel={Tasa de verdaderos positivos}, grid=major,
 legend style={at={(0.98,0.02)}, anchor=south east, font=\footnotesize}]
\addplot[dashed, gray, forget plot] coordinates {(0,0) (1,1)};
""" + "\n".join(curvas) + r"""
\end{axis}\end{tikzpicture}
\caption{""" + TITULOS[evidencia.nombre] + r""": curvas ROC en test de los cinco modelos elegidos; clase positiva: """ + tex(_nombre_clase(evidencia, positiva)) + r""". La diagonal corresponde a un clasificador aleatorio.}
\label{fig:roc-""" + evidencia.nombre + r"""}
\end{figure}"""


def figura_permutacion(evidencia: Evidencia, algoritmo: str, maximo: int = 9) -> str:
    filas = evidencia.permutacion[algoritmo][:maximo][::-1]
    coordenadas = " ".join(f"({numero(f['caida_f1_macro_media']):.4f},{i}) +- ({numero(f['caida_f1_macro_std']):.4f},0)"
                           for i, f in enumerate(filas))
    etiquetas = ",".join("{" + tex(f["atributo"]) + "}" for f in filas)
    return r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[xbar, width=0.8\linewidth, height=""" + f"{max(4.5, 0.62 * len(filas) + 1.5):.1f}" + r"""cm, bar width=7pt,
 xlabel={Caída media del F1 macro en test al permutar}, ytick={""" + ",".join(map(str, range(len(filas)))) + r"""},
 yticklabels={""" + etiquetas + r"""}, xmajorgrids, tick label style={font=\small}, enlarge y limits=0.08,
 scaled x ticks=false, xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}]
\addplot[fill=green!45!black!60, draw=green!40!black, error bars/.cd, x dir=both, x explicit] coordinates {""" + coordenadas + r"""};
\end{axis}\end{tikzpicture}
\caption{""" + TITULOS[evidencia.nombre] + r""": importancia por permutación del modelo """ + tex(algoritmo) + r""" (media y desviación de 10 permutaciones de cada variable en test). Es una herramienta de interpretación: no participó en la selección.}
\label{fig:perm-""" + evidencia.nombre + r"""}
\end{figure}"""


def figura_seleccion(evidencia: Evidencia) -> str:
    completos, reducidos = [], []
    for algoritmo in ALGORITMOS:
        filas = _filas_familia(evidencia, algoritmo)
        sin = _ganadora_configuracion(evidencia, algoritmo)
        con = next(f for f in filas if f["etapa"] == "seleccion_atributos")
        completos.append(f"({algoritmo},{numero(sin['puntuacion_validacion']):.4f})")
        reducidos.append(f"({algoritmo},{numero(con['puntuacion_validacion']):.4f})")
    k = evidencia.experimento["seleccion_atributos"]["k"]
    total = len(evidencia.experimento["seleccion_atributos"]["atributos_transformados"])
    return r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[ybar, bar width=11pt, width=0.9\linewidth, height=5.8cm, ymin=0, ymax=1.15, ytick={0,0.2,...,1},
 ylabel={F1 macro (validación cruzada)}, symbolic x coords={KNN,DT,RF,XGBoost,AdaBoost}, xtick=data,
 enlarge x limits=0.14, ymajorgrids, legend style={at={(0.5,-0.18)},anchor=north,legend columns=2,font=\small},
 nodes near coords, nodes near coords style={font=\tiny, rotate=90, anchor=west, /pgf/number format/fixed, /pgf/number format/precision=3}]
\addplot[fill=blue!50, draw=blue!70!black] coordinates {""" + " ".join(completos) + r"""};
\addplot[fill=teal!45, draw=teal!70!black] coordinates {""" + " ".join(reducidos) + r"""};
\legend{""" + f"Todos los atributos ({total})" + "," + f"Selección por información mutua ({k})" + r"""}
\end{axis}\end{tikzpicture}
\caption{""" + TITULOS[evidencia.nombre] + r""": efecto de la selección de atributos sobre la mejor configuración de cada familia, medido con los mismos cinco pliegues.}
\label{fig:seleccion-""" + evidencia.nombre + r"""}
\end{figure}"""


def seccion_datos(evidencia: Evidencia, procedencia: dict[str, Any], analisis: dict[str, str]) -> str:
    exp = evidencia.experimento
    dataset = evidencia.configuracion["dataset"]
    origen = procedencia[evidencia.nombre]
    derivacion = origen["derivacion"]
    dist = exp["distribuciones"]
    clases = list(dict.fromkeys([*dist["train"], *dist["test"]]))
    filas = [[tex(_nombre_clase(evidencia, c)), str(dist["train"].get(c, 0)), str(dist["test"].get(c, 0)),
              str(dist["train"].get(c, 0) + dist["test"].get(c, 0))] for c in clases]
    filas.append([r"\textbf{Total}", str(exp["muestra_train"]), str(exp["muestra_test"]),
                  str(exp["muestra_train"] + exp["muestra_test"])])
    mayoritaria, proporcion, f1_base = _referencia_mayoritaria(evidencia)
    partes = [
        r"\subsection{Datos y partición}",
        analisis["datos"],
        tabla(["Elemento", "Valor"], [
            ["Archivo analizado", r"\texttt{" + tex(dataset["archivo"]) + "}"],
            ["Original", r"\texttt{" + tex(origen["original"]["ruta"]) + "}, " + str(derivacion["filas_originales"]) + " filas"],
            ["Filas analizadas", str(dataset["filas"])],
            ["Objetivo", r"\texttt{" + tex(exp["target"]) + "}"],
            ["Predictoras", str(len(exp["features"]))],
            ["Fuente", r"\url{" + dataset["fuente"] + "}"],
            ["SHA-256 analizado", _huella(dataset["sha256"])],
        ], caption=f"Procedencia de {TITULOS[evidencia.nombre]}.", alineacion="lp{0.62\\linewidth}"),
        tabla(["Clase", "Train", "Test", "Total"], filas,
              caption=f"Distribución de clases en {TITULOS[evidencia.nombre]} (partición estratificada 75/25, semilla {exp['semilla']}).",
              alineacion="lrrr"),
        "Una referencia trivial que siempre predice la clase mayoritaria de train ("
        + tex(_nombre_clase(evidencia, mayoritaria)) + ") obtendría en test una exactitud de "
        + pct(proporcion) + " y un F1 macro de " + f"{f1_base:.3f}"
        + ". Cualquier modelo útil debe superar con claridad ese F1 macro, no sólo la exactitud.",
    ]
    return "\n\n".join(partes)


def seccion_familia(evidencia: Evidencia, algoritmo: str, analisis: dict[str, str]) -> str:
    filas = _filas_familia(evidencia, algoritmo)
    estandar = _base_estandar(evidencia, algoritmo)
    tabla_filas = []
    for fila in filas:
        nombre = fila["configuracion"]
        base = nombre.split("+")[0]
        descripcion = DESCRIPCION_VARIANTES.get(base, tex(base))
        if fila["etapa"] == "seleccion_atributos":
            descripcion = "mejor anterior + " + str(evidencia.experimento["seleccion_atributos"]["k"]) + " atributos por información mutua"
        marca = r" $\star$" if verdadero(fila["seleccionado"]) else ""
        delta = numero(fila["puntuacion_validacion"]) - numero(estandar["puntuacion_validacion"])
        tabla_filas.append([
            r"\texttt{" + tex(nombre).replace("+", r"+\allowbreak{}") + "}" + marca,
            descripcion,
            p4(fila["f1_macro_validacion"]) + r" $\pm$ " + f"{numero(fila['puntuacion_validacion_std']):.3f}",
            f"{delta:+.3f}" if fila is not estandar else "--",
            pct(fila["accuracy_validacion"]),
            p4(fila["recall_macro_validacion"]),
        ])
    mejor = evidencia.mejores[algoritmo]
    metricas = mejor["metricas_test"]
    etiquetas, matriz = evidencia.matrices[algoritmo]
    analisis_ganador = mejor["metadatos"]["analisis"]
    matriz_filas = [[tex(_nombre_clase(evidencia, e)), *map(str, fila)] for e, fila in zip(etiquetas, matriz)]
    clase_filas = [[tex(_nombre_clase(evidencia, e)), str(sum(matriz[i])),
                    p4(metricas["precision"]["por_clase"][e]["precision"]),
                    p4(metricas["precision"]["por_clase"][e]["recall"]),
                    p4(metricas["precision"]["por_clase"][e]["f1"])] for i, e in enumerate(etiquetas)]
    resumen = (
        r"\textbf{Modelo final.} Configuración \texttt{" + tex(mejor["metadatos"]["configuracion"]) + "}"
        + (" con selección de atributos" if mejor["metadatos"]["seleccion_atributos_aplicada"] else " con todos los atributos")
        + ", reajustada con las " + str(mejor["muestra_train"]) + " filas de train. F1 macro: "
        + p4(analisis_ganador["metricas_train"]["f1_macro"]) + " en entrenamiento, "
        + p4(mejor["metadatos"]["puntuacion_validacion"]) + " en validación cruzada y "
        + p4(metricas["f1"]["macro"]) + " en test (exactitud " + pct(metricas["accuracy"])
        + ", AUC " + f"{analisis_ganador['roc']['auc']:.3f}" + ")."
    )
    partes = [
        r"\subsection{" + NOMBRES[algoritmo] + "}",
        tabla(["Configuración", "Cambio respecto del estándar", "F1 macro CV", r"$\Delta$", "Exact. CV", "Recall CV"],
              tabla_filas,
              caption=NOMBRES[algoritmo] + " en " + TITULOS[evidencia.nombre]
              + r": validación cruzada estratificada de 5 pliegues sobre train. $\star$ = configuración elegida; $\Delta$ = diferencia de F1 macro frente al estándar.",
              alineacion=r">{\raggedright\arraybackslash}p{0.21\linewidth}>{\raggedright\arraybackslash}p{0.26\linewidth}rrrr",
              tamano=r"\footnotesize"),
        resumen,
        r"\begin{table}[H]\footnotesize\begin{minipage}[t]{0.44\linewidth}\centering"
        + r"\captionof{table}{" + NOMBRES[algoritmo] + r": matriz de confusión en test (filas: real; columnas: predicción).}"
        + _tabular(["Real / pred.", *[tex(_nombre_clase(evidencia, e)) for e in etiquetas]], matriz_filas,
                   "l" + "r" * len(etiquetas))
        + r"\end{minipage}\hfill\begin{minipage}[t]{0.52\linewidth}\centering"
        + r"\captionof{table}{" + NOMBRES[algoritmo] + r": métricas por clase en test.}"
        + _tabular(["Clase", "Soporte", "Precisión", "Recall", "F1"], clase_filas, "lrrrr")
        + r"\end{minipage}\end{table}",
        r"\paragraph{Análisis.} " + analisis[algoritmo],
        r"\FloatBarrier",
    ]
    if algoritmo == "DT" and "arbol" in analisis_ganador:
        arbol = analisis_ganador["arbol"]
        partes.insert(-2, r"\begin{figure}[H]\centering\begin{minipage}{0.92\linewidth}\scriptsize\begin{verbatim}" + "\n"
                      + arbol["texto"].rstrip() + "\n" + r"\end{verbatim}\end{minipage}"
                      + r"\caption{Primeros tres niveles del árbol final (profundidad total " + str(arbol["profundidad"])
                      + ", " + str(arbol["hojas"]) + r" hojas). Los umbrales están en las unidades originales: la estandarización se invirtió con la media y desviación de train. $n$ es la cantidad de observaciones de train en el nodo.}\end{figure}")
    return "\n\n".join(partes)


def seccion_sintesis(evidencia: Evidencia, analisis: dict[str, str]) -> str:
    filas = []
    for algoritmo in ALGORITMOS:
        mejor = evidencia.mejores[algoritmo]
        positiva = mejor["metadatos"]["analisis"]["roc"]["clase_positiva"]
        filas.append([
            tex(algoritmo),
            r"\texttt{" + tex(mejor["metadatos"]["configuracion"]) + "}" + (" + sel." if mejor["metadatos"]["seleccion_atributos_aplicada"] else ""),
            p4(mejor["metadatos"]["analisis"]["metricas_train"]["f1_macro"]),
            p4(mejor["metadatos"]["puntuacion_validacion"]),
            p4(mejor["metricas_test"]["f1"]["macro"]),
            pct(mejor["metricas_test"]["accuracy"]),
            p4(mejor["metricas_test"]["precision"]["por_clase"][str(positiva)]["recall"]),
            f"{mejor['metadatos']['analisis']['roc']['auc']:.3f}",
        ])
    elegido = evidencia.experimento["mejor_algoritmo_por_validacion"]
    positiva = evidencia.mejores["RF"]["metadatos"]["analisis"]["roc"]["clase_positiva"]
    return "\n\n".join([
        r"\subsection{Comparación de los cinco modelos elegidos}",
        tabla(["Familia", "Configuración", "F1 train", "F1 CV", "F1 test", "Exactitud test",
               "Recall " + tex(_nombre_clase(evidencia, positiva)), "AUC"], filas,
              caption=TITULOS[evidencia.nombre] + ": resumen de los ganadores. La selección usó sólo la columna F1 CV; el mejor algoritmo por validación es "
              + tex(elegido) + ".", alineacion="llrrrrrr", tamano=r"\footnotesize"),
        figura_cv_test(evidencia),
        figura_roc(evidencia),
        figura_seleccion(evidencia),
        figura_permutacion(evidencia, elegido),
        analisis["sintesis"],
        r"\FloatBarrier",
    ])


def seccion_metodologia(evidencias: list[Evidencia]) -> str:
    from framework_ia.utils import configuraciones_experimento

    catalogo = configuraciones_experimento()
    filas = []
    for algoritmo, configs in catalogo.items():
        for config in configs:
            parametros = ", ".join(f"{k}={v}" for k, v in config["parametros"].items()) or "--"
            filas.append([tex(algoritmo), r"\texttt{" + tex(config["nombre"]) + "}", r"\texttt{" + tex(parametros) + "}",
                          DESCRIPCION_VARIANTES[config["nombre"]]])
    protocolo = []
    for evidencia in evidencias:
        exp = evidencia.experimento
        protocolo.append([tex(TITULOS[evidencia.nombre]), str(exp["muestra_train"] + exp["muestra_test"]),
                          str(exp["muestra_train"]), str(exp["muestra_test"]),
                          "/".join(map(str, exp["muestra_validacion_por_pliegue"])),
                          str(len(exp["features"])),
                          str(len(exp["seleccion_atributos"]["atributos_transformados"])),
                          str(exp["seleccion_atributos"]["k"])])
    metodologia = (RECURSOS / "analisis" / "metodologia.tex").read_text(encoding="utf-8")
    return "\n\n".join([
        metodologia,
        tabla(["Familia", "Nombre", "Parámetros solicitados", "Qué cambia"], filas,
              caption="Catálogo de configuraciones. Cada variante cambia un único aspecto respecto del estándar, para poder atribuirle su efecto.",
              alineacion="llp{0.27\\linewidth}p{0.36\\linewidth}", etiqueta="tab:catalogo", tamano=r"\footnotesize"),
        tabla(["Dataset", "Filas", "Train", "Test", "Validación por pliegue", "Predictoras", "Atributos transf.", "$k$ sel."],
              protocolo, caption="Tamaños efectivos del protocolo (semilla 42).", alineacion="lrrrcrrr", tamano=r"\footnotesize"),
        r"\FloatBarrier",
    ])


def _apendice(evidencias: list[Evidencia], procedencia: dict[str, Any]) -> str:
    partes = [r"\appendix", r"\section{Reproducción de los experimentos}",
              (RECURSOS / "analisis" / "reproduccion.tex").read_text(encoding="utf-8")]
    versiones = {**evidencias[0].configuracion.get("versiones", {}), **evidencias[0].experimento.get("versiones", {})}
    partes.append(tabla(["Paquete", "Versión"], [[tex(k), tex(v)] for k, v in versiones.items()],
                        caption="Versiones registradas durante la ejecución.", alineacion="ll"))
    huellas = []
    for evidencia in evidencias:
        origen = procedencia[evidencia.nombre]
        huellas.append([tex(origen["original"]["ruta"]), _huella(origen["original"]["sha256"])])
        huellas.append([tex(origen["ruta"]), _huella(origen["sha256"])])
    partes.append(tabla(["Archivo", "SHA-256"], huellas, caption="Huellas de los datos utilizados.",
                        alineacion="lp{0.6\\linewidth}"))
    return "\n\n".join(partes)


PREAMBULO = r"""\documentclass[11pt]{article}
\usepackage[spanish,es-nodecimaldot,es-noshorthands]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{booktabs,longtable,tabularx,amsmath,amssymb,float,xurl,placeins,xcolor}
\usepackage{pgfplots}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc,babel}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}
\usepackage[a4paper,margin=2.3cm]{geometry}
\pgfplotsset{compat=1.18}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.12}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
"""


def construir_documento(evidencias: list[Evidencia], autores: list[str], procedencia: dict[str, Any], fecha: str) -> str:
    portada = (
        r"\title{\textbf{Laboratorio 2}\\[2pt] Métodos supervisados: clasificación\\[2pt]\large KNN, árboles de decisión, Random Forest, XGBoost y AdaBoost}" + "\n"
        + r"\author{" + r"\\".join(tex(a) for a in autores) + "}\n"
        + r"\date{EIF420O -- Inteligencia Artificial, I Ciclo 2026\\Universidad Nacional de Costa Rica\\Profesor: Dr. Juan Murillo-Morera\\" + tex(fecha) + "}\n"
    )
    documento = [PREAMBULO, portada, r"\begin{document}", r"\maketitle", r"\begin{abstract}",
                 (RECURSOS / "analisis" / "resumen.tex").read_text(encoding="utf-8"), r"\end{abstract}",
                 r"\tableofcontents", r"\clearpage",
                 (RECURSOS / "analisis" / "introduccion.tex").read_text(encoding="utf-8"),
                 (RECURSOS / "fundamentos.tex").read_text(encoding="utf-8"),
                 r"\clearpage", seccion_metodologia(evidencias)]
    for evidencia in evidencias:
        analisis = bloques_analisis(evidencia.nombre)
        faltan = {"datos", "sintesis", *ALGORITMOS} - set(analisis)
        if faltan:
            raise ValueError(f"Faltan bloques de análisis para {evidencia.nombre}: {sorted(faltan)}")
        documento += [r"\clearpage", r"\section{Resultados: " + TITULOS[evidencia.nombre] + "}",
                      seccion_datos(evidencia, procedencia, analisis)]
        documento += [seccion_familia(evidencia, a, analisis) for a in ALGORITMOS]
        documento.append(seccion_sintesis(evidencia, analisis))
    documento += [r"\clearpage", (RECURSOS / "analisis" / "discusion.tex").read_text(encoding="utf-8"),
                  (RECURSOS / "analisis" / "conclusiones.tex").read_text(encoding="utf-8"),
                  r"\clearpage", (RECURSOS / "referencias.tex").read_text(encoding="utf-8"),
                  r"\clearpage", _apendice(evidencias, procedencia), r"\end{document}"]
    return "\n\n".join(documento) + "\n"


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Informe LAB 02 desde resultados verificados; no entrena.")
    parser.add_argument("resultados", type=Path, help="Carpeta con potabilidad/ y diabetes/.")
    parser.add_argument("--salida", type=Path, default=Path("output/lab02/informe"))
    parser.add_argument("--autor", action="append", default=[], help="Nombre completo (ID); repetir por autor.")
    parser.add_argument("--fecha", default="15 de octubre de 2026")
    parser.add_argument("--procedencia", type=Path, default=RAIZ / "data" / "lab02" / "procedencia.json")
    return parser


def ejecutar(args: argparse.Namespace) -> Path:
    evidencias = [cargar_evidencia(nombre, args.resultados / nombre) for nombre in ("potabilidad", "diabetes")]
    procedencia = _json(args.procedencia)
    for evidencia in evidencias:
        if procedencia[evidencia.nombre]["sha256"] != evidencia.configuracion["dataset"]["sha256"]:
            raise ValueError(f"{evidencia.nombre}: la ejecución no corresponde a los datos registrados.")
    documento = construir_documento(evidencias, args.autor or ["Autores no especificados"], procedencia, args.fecha)
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta = args.salida / "main.tex"
    ruta.write_text(documento, encoding="utf-8", newline="\n")
    manifest = {
        "autores": args.autor,
        "fecha_documento": args.fecha,
        "fuentes": {e.nombre: {"dataset_sha256": e.configuracion["dataset"]["sha256"],
                               "archivos_resultados_sha256": e.hashes} for e in evidencias},
        "main_tex_sha256": _hash(ruta),
    }
    (args.salida / "manifest_informe.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.salida / "README.md").write_text(
        "# Proyecto Overleaf de LAB 02\n\nImporte `proyecto_overleaf.zip` en Overleaf (New Project > Upload Project) "
        "y compile `main.tex` con pdfLaTeX. Tablas, figuras PGFPlots y referencias están dentro del archivo: "
        "no necesita imágenes ni BibTeX.\n", encoding="utf-8")
    with zipfile.ZipFile(args.salida / "proyecto_overleaf.zip", "w", compression=zipfile.ZIP_DEFLATED) as archivo:
        for nombre in ("main.tex", "README.md", "manifest_informe.json"):
            archivo.write(args.salida / nombre, arcname=nombre)
    return ruta


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    args = parser.parse_args(argv)
    try:
        ruta = ejecutar(args)
    except (ValueError, KeyError, TypeError, OSError, StopIteration, json.JSONDecodeError) as exc:
        parser.exit(2, f"No se pudo generar el informe: {exc}\n")
    print(f"Informe LaTeX y proyecto Overleaf guardados en {ruta.parent}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
