"""Construye un informe LaTeX autónomo a partir de evidencia real de LAB 02.

No entrena, modifica datasets, descarga ni publica. Comprueba la coherencia de
particiones, selección y matrices antes de escribir. Las figuras PGFPlots y la
bibliografía quedan dentro de main.tex, que se puede compilar en Overleaf sin
archivos externos.
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
NOMBRES = {"KNN": "KNN", "DT": "Árbol de decisión", "RF": "Random Forest",
           "XGBoost": "XGBoost", "AdaBoost": "AdaBoost"}
PARAMETROS = {
    "KNN": ("n_neighbors", "weights", "p", "metric"),
    "DT": ("criterion", "max_depth", "min_samples_split", "min_samples_leaf"),
    "RF": ("n_estimators", "criterion", "max_depth", "min_samples_leaf", "max_features"),
    "XGBoost": ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"),
    "AdaBoost": ("n_estimators", "learning_rate", "estimator", "algorithm"),
}
PREGUNTAS = {
    "KNN": "El número de vecinos, sus pesos y la distancia cambian la decisión local. La comparación mantiene el mismo escalado y las mismas filas para aislar el procedimiento de evaluación.",
    "DT": "La profundidad y los mínimos de muestras controlan la complejidad del árbol. Una puntuación de validación mayor no demuestra por sí sola que el árbol haya dejado de sobreajustar.",
    "RF": "Número de árboles, atributos por división y límites de profundidad cambian el ensamble. La comparación no permite atribuir una diferencia a un único parámetro cuando se cambian varios simultáneamente.",
    "XGBoost": "Profundidad, tasa de aprendizaje y número de árboles interactúan. El experimento utiliza XGBClassifier real y no sustituye esta familia por GradientBoostingClassifier.",
    "AdaBoost": "Tasa de aprendizaje y número de estimadores regulan la combinación secuencial. Un mayor número de estimadores no implica una mejora universal y puede enfatizar observaciones ruidosas.",
}
PREAMBULO = r"""\documentclass[11pt]{article}
\usepackage[spanish,es-nodecimaldot,es-noshorthands]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs,longtable,tabularx,graphicx,amsmath,float,xurl}
\usepackage{pgfplots,placeins}
\usepackage[hidelinks]{hyperref}
\usepackage[a4paper,margin=2.5cm]{geometry}
\pgfplotsset{compat=1.18}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.12}
\setlength{\parskip}{0.35em}
"""


@dataclass
class Evidencia:
    nombre: str
    ruta: Path
    configuracion: dict[str, Any]
    comparacion: list[dict[str, Any]]
    mejores: dict[str, Any]
    matrices: dict[str, tuple[list[str], list[list[int]]]]
    hashes: dict[str, str]

    @property
    def experimento(self) -> dict[str, Any]:
        return self.configuracion["experimento"]


def tex(valor: Any) -> str:
    """Escapa texto de JSON/CSV, sin interpretarlo como código LaTeX."""
    mapa = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
            "$": r"\$", "#": r"\#", "_": r"\_\allowbreak{}",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(mapa.get(c, c) for c in str(valor)).replace("\n", " ")


def url(valor: Any) -> str:
    valor = str(valor)
    if any(c in valor for c in "{}\\\n\r"):
        return tex(valor)
    return r"\url{" + valor + "}"


def numero(valor: Any) -> float:
    resultado = float(valor)
    if not math.isfinite(resultado) or not 0 <= resultado <= 1:
        raise ValueError(f"Métrica fuera de [0,1] o no finita: {valor!r}")
    return resultado


def porcentaje(valor: Any) -> str:
    return f"{100 * numero(valor):.2f}" + r"\%"


def puntuacion(valor: Any) -> str:
    return f"{numero(valor):.4f}"


def verdadero(valor: Any) -> bool:
    return str(valor).lower() in ("true", "1")


def _json(ruta: Path) -> dict[str, Any]:
    def rechazar_constante(valor: str) -> None:
        raise ValueError(f"JSON no estándar en {ruta.name}: {valor}")

    contenido = json.loads(ruta.read_text(encoding="utf-8-sig"),
                           parse_constant=rechazar_constante)
    if not isinstance(contenido, dict):
        raise TypeError(f"Se esperaba un objeto JSON: {ruta}")
    return contenido


def _hash(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _huella_tex(huella: str) -> str:
    return (r"{\footnotesize\texttt{" + huella[:32]
            + r"}\newline\texttt{" + huella[32:] + "}}")


def _validar_derivacion_diabetes(evidencia: Evidencia, origen: dict[str, Any]) -> None:
    """Impide presentar resultados de otra entrada como la nueva derivación."""
    original, derivacion = origen.get("original", {}), origen.get("derivacion", {})
    regla = derivacion.get("regla", {})
    if regla != {"columna": "glyhb", "operador": ">", "umbral": 7.0,
                 "positiva": "Diabetes", "negativa": "No diabetes"}:
        raise ValueError("Diabetes requiere la regla documentada glyhb > 7 y su procedencia.")
    filas_originales = derivacion.get("filas_originales")
    filas_preparadas = derivacion.get("filas_preparadas")
    excluidos = derivacion.get("excluidos_sin_glyhb", [])
    if (filas_originales != original.get("filas") or filas_originales != 403
            or filas_preparadas != evidencia.configuracion["dataset"]["filas"]
            or filas_preparadas != 390 or len(excluidos) != 13):
        raise ValueError("La derivación de Diabetes no acredita las 403 filas originales y 390 etiquetadas.")
    if filas_originales - len(excluidos) != filas_preparadas:
        raise ValueError("Las exclusiones de glyhb no explican la muestra preparada.")
    indices = [fila["fila_original"] for fila in excluidos]
    if len(set(indices)) != len(indices) or any(not 0 <= i < filas_originales for i in indices):
        raise ValueError("Índices de exclusión de Diabetes duplicados o fuera de rango.")
    predictoras = derivacion.get("predictoras", [])
    nativas = {"chol", "stab.glu", "hdl", "ratio", "location", "age", "gender",
               "height", "weight", "frame", "bp.1s", "bp.1d", "bp.2s", "bp.2d",
               "waist", "hip", "time.ppn"}
    if (len(predictoras) != 17 or set(predictoras) != nativas
            or predictoras != evidencia.experimento["features"]):
        raise ValueError("Diabetes debe usar las 17 predictoras nativas, excluyendo id y glyhb.")
    if set(derivacion.get("categoricas", [])) != {"location", "gender", "frame"}:
        raise ValueError("La derivación no conserva las tres categóricas nativas de Diabetes.")
    if not evidencia.experimento["preprocesamiento"]["incluir_categoricas"]:
        raise ValueError("El experimento debe incluir las categóricas nativas de Diabetes.")
    trazabilidad = derivacion.get("trazabilidad", {})
    if trazabilidad.get("filas") != filas_preparadas:
        raise ValueError("La trazabilidad no cubre las 390 filas preparadas de Diabetes.")
    for entrada in (original, trazabilidad):
        if not re.fullmatch(r"[a-fA-F0-9]{64}", entrada.get("sha256", "")):
            raise ValueError("Falta SHA-256 del original o la trazabilidad de Diabetes.")
    if derivacion.get("lectura") != {"separador": ",", "decimal": ".", "encoding": "utf-8"}:
        raise ValueError("La preparación de Diabetes debe conservar lectura coma/punto/UTF-8.")


def cargar_evidencia(nombre: str, ruta: Path, *, permitir_parcial: bool = False) -> Evidencia:
    requeridos = ("configuracion.json", "comparacion_validacion.csv", "mejores_modelos.json")
    for archivo in requeridos:
        if not (ruta / archivo).is_file():
            raise ValueError(f"Falta evidencia real para {nombre}: {ruta / archivo}")
    configuracion = _json(ruta / requeridos[0])
    mejores = _json(ruta / requeridos[2])
    with (ruta / requeridos[1]).open(encoding="utf-8-sig", newline="") as entrada:
        comparacion = list(csv.DictReader(entrada))
    if not comparacion or not mejores:
        raise ValueError(f"No hay resultados experimentales para {nombre}.")
    experimento = configuracion["experimento"]
    if experimento.get("test_usado_para_seleccion") is not False:
        raise ValueError("El informe exige test excluido de la selección.")
    if experimento.get("reajustado_train_validacion") is not False:
        raise ValueError("Esta metodología requiere ganadores ajustados sólo en train.")
    if experimento.get("preprocesamiento", {}).get("ajustado_solo_train") is not True:
        raise ValueError("No se acredita preparación ajustada sólo en train.")
    huella = configuracion["dataset"].get("sha256", "")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", huella):
        raise ValueError("Falta un SHA-256 válido de la entrada.")
    indices = [experimento.get(f"{particion}_indices", [])
               for particion in ("train", "validacion", "test")]
    conjuntos = [{json.dumps(i, sort_keys=True) for i in lista} for lista in indices]
    if any(not lista or len(lista) != len(conjunto)
           for lista, conjunto in zip(indices, conjuntos)):
        raise ValueError("Particiones vacías o índices duplicados.")
    if any(conjuntos[i] & conjuntos[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Hay solapamiento entre train, validación y test.")
    for particion, lista in zip(("train", "validacion", "test"), indices):
        if len(lista) != experimento[f"muestra_{particion}"]:
            raise ValueError(f"Tamaño incompatible de {particion}.")
    if sum(map(len, indices)) != configuracion["dataset"]["filas"] - experimento.get("filas_objetivo_nulo", 0):
        raise ValueError("Las particiones no explican todas las filas válidas.")
    if set(mejores) != set(ALGORITMOS) and not permitir_parcial:
        raise ValueError("El informe completo requiere ganadores de las cinco familias.")
    if not set(mejores) <= set(ALGORITMOS):
        raise ValueError("Se encontraron familias ajenas a LAB 02.")
    matrices = {}
    hashes = {archivo: _hash(ruta / archivo) for archivo in requeridos}
    for algoritmo, mejor in mejores.items():
        filas = [f for f in comparacion if f["algoritmo"] == algoritmo]
        if not filas or not any(f["configuracion"] == "estandar" for f in filas):
            raise ValueError(f"Falta la configuración estándar de {algoritmo}.")
        if len(filas) < 2 and not permitir_parcial:
            raise ValueError(f"Faltan variantes de {algoritmo}.")
        seleccionadas = [f for f in filas if verdadero(f["seleccionado"])]
        exitosas = [f for f in filas if f["estado"] == "ok"]
        if len(seleccionadas) != 1 or not exitosas:
            raise ValueError(f"Selección ambigua de {algoritmo}.")
        seleccionada = seleccionadas[0]
        if seleccionada["estado"] != "ok":
            raise ValueError(f"Se seleccionó un candidato fallido de {algoritmo}.")
        if not math.isclose(numero(seleccionada["puntuacion_validacion"]),
                            max(numero(f["puntuacion_validacion"]) for f in exitosas),
                            abs_tol=1e-12):
            raise ValueError(f"El ganador de {algoritmo} no maximiza validación.")
        if seleccionada["configuracion"] != experimento["selecciones"][algoritmo]:
            raise ValueError(f"Selección de {algoritmo} incompatible con metadatos.")
        if seleccionada["configuracion"] != mejor["metadatos"]["configuracion"]:
            raise ValueError(f"El modelo de test no es el ganador de {algoritmo}.")
        ruta_matriz = ruta / algoritmo / "matriz_confusion_test.csv"
        with ruta_matriz.open(encoding="utf-8-sig", newline="") as entrada:
            contenido = list(csv.reader(entrada))
        etiquetas = contenido[0][1:]
        if [f[0] for f in contenido[1:]] != etiquetas:
            raise ValueError(f"Etiquetas incompatibles de matriz de {algoritmo}.")
        matriz = [[int(c) for c in f[1:]] for f in contenido[1:]]
        if len(etiquetas) < 2 or any(len(f) != len(etiquetas) for f in matriz):
            raise ValueError(f"Matriz no cuadrada de {algoritmo}.")
        if any(c < 0 for fila in matriz for c in fila):
            raise ValueError(f"Conteos negativos en {algoritmo}.")
        total = sum(map(sum, matriz))
        if total != mejor["muestra_test"] or total != experimento["muestra_test"]:
            raise ValueError(f"Matriz de {algoritmo} no coincide con tamaño de test.")
        accuracy = sum(matriz[i][i] for i in range(len(matriz))) / total
        if not math.isclose(accuracy, numero(mejor["metricas_test"]["accuracy"]), abs_tol=1e-10):
            raise ValueError(f"Accuracy de {algoritmo} no coincide con la matriz.")
        metricas = mejor["metricas_test"]["precision"]["por_clase"]
        f1s = []
        for i, etiqueta in enumerate(etiquetas):
            soporte = sum(matriz[i])
            predichas = sum(f[i] for f in matriz)
            tp = matriz[i][i]
            precision = tp / predichas if predichas else 0.0
            recall = tp / soporte if soporte else 0.0
            f1 = 2 * tp / (soporte + predichas) if soporte + predichas else 0.0
            f1s.append(f1)
            for clave, esperado in (("precision", precision), ("recall", recall), ("f1", f1)):
                if not math.isclose(esperado, numero(metricas[etiqueta][clave]), abs_tol=1e-10):
                    raise ValueError(f"{clave} de {algoritmo}/{etiqueta} no coincide con matriz.")
        if not math.isclose(sum(f1s) / len(f1s), numero(mejor["metricas_test"]["f1"]["macro"]), abs_tol=1e-10):
            raise ValueError(f"F1 macro de {algoritmo} no coincide con matriz.")
        matrices[algoritmo] = (etiquetas, matriz)
        hashes[f"{algoritmo}/matriz_confusion_test.csv"] = _hash(ruta_matriz)
    global_elegido = experimento["mejor_algoritmo_por_validacion"]
    if global_elegido not in mejores and not permitir_parcial:
        raise ValueError("El ganador global no tiene evaluación final.")
    if global_elegido in mejores:
        todas = [f for f in comparacion if f["estado"] == "ok"]
        mejor_fila = next(f for f in todas if f["algoritmo"] == global_elegido and verdadero(f["seleccionado"]))
        if not math.isclose(numero(mejor_fila["puntuacion_validacion"]),
                            max(numero(f["puntuacion_validacion"]) for f in todas), abs_tol=1e-12):
            raise ValueError("El ganador global no maximiza la métrica de validación.")
    return Evidencia(nombre, ruta, configuracion, comparacion, mejores, matrices, hashes)


def tabla(encabezados: list[str], filas: list[list[str]], *, caption: str,
          alineacion: str | None = None) -> str:
    alineacion = alineacion or ("l" + "r" * (len(encabezados) - 1))
    contenido = [r"\begin{table}[H]\centering\small", r"\caption{" + caption + "}",
                 r"\begin{tabular}{" + alineacion + r"}\toprule",
                 " & ".join(encabezados) + r" \\ \midrule"]
    contenido.extend(" & ".join(fila) + r" \\" for fila in filas)
    contenido.extend((r"\bottomrule\end{tabular}", r"\end{table}"))
    return "\n".join(contenido)


def pares(parejas: list[tuple[str, str]], caption: str) -> str:
    resultado = [r"\begin{longtable}{p{0.31\linewidth}p{0.61\linewidth}}",
                 r"\caption{" + caption + r"}\\\toprule",
                 r"Elemento & Valor \\ \midrule\endfirsthead",
                 r"\toprule Elemento & Valor \\ \midrule\endhead"]
    resultado.extend(tex(k) + " & " + v + r" \\" for k, v in parejas)
    resultado.append(r"\bottomrule\end{longtable}")
    return "\n".join(resultado)


def _parametros(fila: dict[str, Any]) -> dict[str, Any]:
    valor = fila.get("parametros", {})
    return json.loads(valor) if isinstance(valor, str) else valor


def fila_seleccionada(evidencia: Evidencia, algoritmo: str) -> dict[str, Any]:
    return next(f for f in evidencia.comparacion
                if f["algoritmo"] == algoritmo and verdadero(f["seleccionado"]))


def grafico_resumen(evidencia: Evidencia) -> str:
    algoritmos = [a for a in ALGORITMOS if a in evidencia.mejores]
    validacion = " ".join(f"({i},{numero(fila_seleccionada(evidencia, a)['f1_macro_validacion']):.8f})"
                          for i, a in enumerate(algoritmos))
    prueba = " ".join(f"({i},{numero(evidencia.mejores[a]['metricas_test']['f1']['macro']):.8f})"
                       for i, a in enumerate(algoritmos))
    return r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[ybar,bar width=9pt,width=0.96\linewidth,height=6cm,
ymin=0,ymax=1.05,ylabel={F1 macro},xtick={""" + ",".join(map(str, range(len(algoritmos)))) + r"""},
xticklabels={""" + ",".join(algoritmos) + r"""},
enlarge x limits=0.13,grid=major,tick label style={font=\small},
legend style={at={(0.5,-0.18)},anchor=north,legend columns=2,font=\small}]
\addplot[fill=blue!55,draw=blue!70!black] coordinates {""" + validacion + r"""};
\addplot[fill=orange!65,draw=orange!80!black] coordinates {""" + prueba + r"""};
\legend{Validación,Test}
\end{axis}\end{tikzpicture}
\caption{Ganadores por familia en """ + tex(evidencia.nombre.capitalize()) + r""".
Las barras corresponden a muestras distintas; su diferencia es descriptiva.
La selección se realizó antes de evaluar test.}
\end{figure}"""


def _descripcion_datos(evidencia: Evidencia, procedencia: dict[str, Any]) -> str:
    config, exp = evidencia.configuracion, evidencia.experimento
    entrada = config["dataset"]
    origen = procedencia.get(evidencia.nombre, {})
    if origen.get("sha256") and origen["sha256"] != entrada["sha256"]:
        raise ValueError(f"Procedencia y ejecución usan hashes distintos en {evidencia.nombre}.")
    fuente = entrada.get("fuente") or origen.get("fuente")
    parejas = [("Archivo", tex(entrada["archivo"])), ("Filas de entrada", str(entrada["filas"])),
               ("Objetivo", r"\texttt{" + tex(exp["target"]) + "}"),
               ("Predictoras", tex(", ".join(exp["features"]))),
               ("Fuente registrada", url(fuente) if fuente else "No declarada en la ejecución."),
               ("SHA-256 del CSV", _huella_tex(entrada["sha256"])),
               ("Filas con objetivo ausente", str(exp.get("filas_objetivo_nulo", 0)))]
    resultado = [r"\section{Datos y resultados: " + tex(evidencia.nombre.capitalize()) + "}",
                 pares(parejas, "Procedencia y alcance de la entrada experimental.")]
    if evidencia.nombre == "potabilidad":
        resultado.append(r"El CSV público de Water Quality conserva la etiqueta original \texttt{Potability}: 0 significa no potable y 1 potable. El caso de clase usa una versión de 2011 filas con etiquetas textuales. Esta ejecución utiliza las filas de entrada indicadas arriba y no se presenta como una reproducción idéntica de aquel archivo. Las coincidencias de esquema y observaciones apoyan una relación entre fuentes, pero no acreditan igualdad íntegra con la versión del profesor. Las ausencias de predictoras se tratan dentro del entrenamiento; no se confunde esa imputación con un filtrado previo de casos completos.")
    if evidencia.nombre == "diabetes":
        _validar_derivacion_diabetes(evidencia, origen)
        original, derivacion = origen["original"], origen["derivacion"]
        traza = derivacion["trazabilidad"]
        resultado.append(r"El dataset original contiene 403 observaciones y 19 columnas, sin una etiqueta categórica de diabetes. La documentación primaria del conjunto describe el criterio histórico de hemoglobina glicosilada mayor que 7.0 \cite{datosdiabetes}. Se utiliza esa regla para construir la etiqueta experimental: \texttt{glyhb}>7 produce \texttt{Diabetes}; \texttt{glyhb}$\leq$7 produce \texttt{No diabetes}. Se emplea como regla experimental histórica del conjunto y no como criterio diagnóstico clínico actual.")
        resultado.append(r"Se excluyen únicamente las 13 filas cuyo \texttt{glyhb} está ausente: no se imputa un objetivo desconocido. Quedan 390 observaciones etiquetadas. Se retiran \texttt{id}, por ser identificador, y \texttt{glyhb}, porque define la respuesta y conservarlo como predictor introduciría fuga directa. Las 17 predictoras son columnas nativas del original: 14 numéricas y las categóricas \texttt{location}, \texttt{gender} y \texttt{frame}. No se calcula BMI ni una nueva razón cintura/cadera. La columna \texttt{ratio} ya pertenece a la entrada.")
        resultado.append(r"Los valores ausentes de las predictoras se conservan durante esta preparación; su imputación por mediana o moda se aprende después, únicamente con train. En particular, \texttt{bp.2s} y \texttt{bp.2d} mantienen 252 ausencias cada una entre las 390 filas etiquetadas. El archivo preparado usa coma como separador, punto decimal y UTF-8. Se conserva una tabla de correspondencia entre fila preparada, fila original e ID original, con índices de fila desde cero. La documentación y las coincidencias de esquema no prueban identidad completa con el archivo adaptado del material de clase.")
        resultado.append(pares([
            ("Original preservado", tex(original["ruta"])),
            ("SHA-256 original", _huella_tex(original["sha256"])),
            ("Filas originales / preparadas", str(derivacion["filas_originales"]) + " / " + str(derivacion["filas_preparadas"])),
            ("Script de preparación", tex(derivacion["script"])),
            ("Correspondencia de filas", tex(traza["ruta"]) + "; " + str(traza["filas"]) + " registros."),
            ("SHA-256 correspondencia", _huella_tex(traza["sha256"])),
        ], "Trazabilidad de la derivación del objetivo de Diabetes."))
        resultado.append(tabla(["Fila original (base 0)", "ID original"],
                               [[str(f["fila_original"]), tex(f["id_original"])]
                                for f in derivacion["excluidos_sin_glyhb"]],
                               caption="Observaciones excluidas por ausencia de glyhb; sin imputación del objetivo.",
                               alineacion="rr"))
    distribuciones = exp["distribuciones"]
    clases = list(dict.fromkeys(c for d in distribuciones.values() for c in d))
    filas = []
    for clase in clases:
        conteos = [int(distribuciones[particion].get(clase, 0))
                   for particion in ("train", "validacion", "test")]
        filas.append([tex(clase), *map(str, conteos), str(sum(conteos))])
    filas.append([r"\textbf{Total}", *[str(exp[f"muestra_{p}"]) for p in ("train", "validacion", "test")],
                  str(sum(exp[f"muestra_{p}"] for p in ("train", "validacion", "test")))])
    resultado.append(tabla(["Clase real", "Train", "Validación", "Test", "Total"], filas,
                           caption="Distribución observada de etiquetas en particiones disjuntas."))
    mayoritaria = max(distribuciones["train"], key=distribuciones["train"].get)
    proporcion = distribuciones["test"].get(mayoritaria, 0) / exp["muestra_test"]
    f1_base = 2 * proporcion / (1 + proporcion) / len(clases)
    resultado.append("Una referencia que predice siempre la clase mayoritaria de train, " + tex(mayoritaria)
                     + ", obtiene en test accuracy " + porcentaje(proporcion) + " y F1 macro "
                     + puntuacion(f1_base) + ". La clase se fija usando train; el valor de test se calcula a partir de sus conteos. Esta referencia permite distinguir aprendizaje útil de favorecer solamente una clase frecuente.")
    resumen = []
    for algoritmo in ALGORITMOS:
        if algoritmo not in evidencia.mejores:
            continue
        fila = fila_seleccionada(evidencia, algoritmo)
        metricas = evidencia.mejores[algoritmo]["metricas_test"]
        resumen.append([tex(algoritmo), tex(fila["configuracion"]),
                        puntuacion(fila["puntuacion_validacion"]),
                        porcentaje(metricas["accuracy"]), puntuacion(metricas["f1"]["macro"])])
    resultado.append(tabla(["Familia", "Configuración", "Score val.", "Acc. test", "F1 test"], resumen,
                           caption="Configuraciones elegidas mediante " + tex(exp["metrica_seleccion"]) + "; prueba final separada.",
                           alineacion="llrrr"))
    resultado.append(grafico_resumen(evidencia))
    elegido = exp["mejor_algoritmo_por_validacion"]
    score_elegido = numero(fila_seleccionada(evidencia, elegido)["puntuacion_validacion"]) if elegido in evidencia.mejores else None
    empate = [a for a in ALGORITMOS if a in evidencia.mejores and score_elegido is not None
              and math.isclose(numero(fila_seleccionada(evidencia, a)["puntuacion_validacion"]),
                               score_elegido, abs_tol=1e-12)]
    interpretacion = "El algoritmo global seleccionado por validación es " + r"\textbf{" + tex(elegido) + "}. "
    if len(empate) > 1:
        interpretacion += ("Existe empate de score de validación entre " + tex(", ".join(empate))
                          + "; se resuelve por el orden declarado de las familias. La elección no acredita superioridad del algoritmo seleccionado frente al otro ganador empatado. ")
    interpretacion += "El orden de las métricas de test no se utiliza para reemplazar esa selección. Un resultado distinto entre conjuntos debe describirse como incertidumbre de generalización, sin volver a optimizar sobre el test reservado."
    resultado.append(interpretacion)
    resultado.append(r"\FloatBarrier")
    return "\n\n".join(resultado)


def _resultado_familia(evidencia: Evidencia, algoritmo: str) -> str:
    filas = [f for f in evidencia.comparacion if f["algoritmo"] == algoritmo]
    seleccion = fila_seleccionada(evidencia, algoritmo)
    mejor = evidencia.mejores[algoritmo]
    metricas = mejor["metricas_test"]
    resultado = [r"\subsection{" + tex(NOMBRES[algoritmo]) + "}", tex(PREGUNTAS[algoritmo])]
    comparacion = []
    for fila in filas:
        nombre = tex(fila["configuracion"]) + (r" $\star$" if verdadero(fila["seleccionado"]) else "")
        comparacion.append([nombre, *[puntuacion(fila[c]) if fila["estado"] == "ok" else "Error"
                                     for c in ("accuracy_validacion", "recall_macro_validacion", "f1_macro_validacion")]])
    resultado.append(tabla(["Configuración", "Acc. val.", "Recall macro", "F1 macro"], comparacion,
                           caption="Comparación de " + tex(algoritmo) + r" en validación. $\star$: configuración elegida."))
    keys = list(PARAMETROS[algoritmo])
    for fila in filas:
        for key in evidencia.experimento["configuraciones"][algoritmo]:
            if key["nombre"] == fila["configuracion"]:
                keys.extend(k for k in key["parametros"] if k not in keys)
    parejas = []
    for fila in filas:
        params = _parametros(fila)
        resumen = "; ".join(f"{k}={params[k]}" for k in keys if k in params)
        parejas.append((fila["configuracion"], tex(resumen) if resumen else "Sin parámetros efectivos por fallo de ajuste."))
    resultado.append(pares(parejas, "Parámetros registrados relevantes; el JSON conserva el conjunto completo."))
    if algoritmo == "XGBoost":
        resultado.append(r"En la configuración estándar, \texttt{None} en el registro de algunas opciones indica que XGBClassifier delega ese valor al comportamiento predeterminado de la versión instalada; no significa ausencia de árboles, profundidad o aprendizaje. Las variantes con valores explícitos y la versión registrada delimitan qué puede reproducirse. El archivo conserva los parámetros expuestos por \texttt{get\_params()}, no una inspección adicional de la configuración interna del booster.")
    estandar = next(f for f in filas if f["configuracion"] == "estandar")
    if estandar["estado"] == "ok":
        delta = numero(seleccion["puntuacion_validacion"]) - numero(estandar["puntuacion_validacion"])
        empatadas = [f["configuracion"] for f in filas if f["estado"] == "ok" and
                    math.isclose(numero(f["puntuacion_validacion"]), numero(seleccion["puntuacion_validacion"]), abs_tol=1e-12)]
        interpretacion = ("Se selecciona " + r"\textbf{" + tex(seleccion["configuracion"]) + "}"
                          + " con score de validación " + puntuacion(seleccion["puntuacion_validacion"])
                          + f"; la diferencia frente al estándar es {delta:+.4f} en la métrica de selección.")
        if len(empatadas) > 1:
            interpretacion += " Hay empate entre " + tex(", ".join(empatadas)) + "; se conserva la primera configuración en el orden declarado. El empate no demuestra superioridad estadística."
        elif seleccion["configuracion"] == "estandar":
            interpretacion += " Las variaciones evaluadas no superaron al estándar bajo este criterio y esta partición."
        else:
            interpretacion += " La variante supera al estándar en esta partición de selección; no establece un óptimo universal ni una diferencia estadísticamente significativa."
        resultado.append(interpretacion)
    else:
        resultado.append("La configuración estándar no pudo evaluarse; la selección sólo compara candidatos exitosos. Esta limitación debe conservarse al interpretar la familia.")
    etiquetas, matriz = evidencia.matrices[algoritmo]
    tabla_matriz = tabla(["Real / pred.", *map(tex, etiquetas)],
                         [[tex(e), *map(str, fila)] for e, fila in zip(etiquetas, matriz)],
                         caption="Matriz de confusión de " + tex(algoritmo) + " en test: filas reales, columnas predichas.")
    resultado.append(tabla_matriz)
    por_clase = metricas["precision"]["por_clase"]
    resultado.append(tabla(["Clase", "Soporte", "Precisión", "Recall", "F1"],
                           [[tex(e), str(sum(matriz[i])), puntuacion(por_clase[e]["precision"]),
                             puntuacion(por_clase[e]["recall"]), puntuacion(por_clase[e]["f1"])]
                            for i, e in enumerate(etiquetas)],
                           caption="Desempeño de test por clase del ganador de " + tex(algoritmo) + "."))
    errores = sum(sum(fila) - fila[i] for i, fila in enumerate(matriz))
    resultado.append("En test, accuracy es " + porcentaje(metricas["accuracy"]) + ", error global "
                     + porcentaje(1 - numero(metricas["accuracy"])) + ", y F1 macro "
                     + puntuacion(metricas["f1"]["macro"]) + f". La matriz registra {errores} errores de "
                     + str(mejor["muestra_test"]) + " observaciones.")
    comentarios = []
    for i, etiqueta in enumerate(etiquetas):
        soporte = sum(matriz[i])
        perdidos = soporte - matriz[i][i]
        falsas = sum(f[i] for f in matriz) - matriz[i][i]
        comentarios.append("Para " + tex(etiqueta) + f", {perdidos} casos reales se predicen como otra clase y {falsas} casos de otras clases se asignan a ella")
    resultado.append("; ".join(comentarios) + ". Estas diferencias distinguen errores de omisión y falsas asignaciones. Deben interpretarse según el significado de las etiquetas y sin atribuir validez clínica o certificación de agua al clasificador.")
    resultado.append(r"\FloatBarrier")
    return "\n\n".join(resultado)


def _metodologia(evidencias: list[Evidencia]) -> str:
    texto = [r"\section{Metodología reproducible}",
             r"El framework separa \texttt{datos}, \texttt{modelos/supervisado}, resultados y visualización. La clase \texttt{Clasificacion} hereda de \texttt{Supervisado}; \texttt{experimentar} ejecuta el mismo flujo desde la API y desde \texttt{scripts.ejecutar\_lab02}. La interfaz no participa en el ajuste de los experimentos de este informe.",
             r"Se separa primero un test estratificado y, dentro del resto, una validación estratificada. Cada candidato utiliza exactamente los mismos índices. La imputación numérica por mediana, la categórica por moda, la codificación one-hot cuando corresponde y el escalado de variables numéricas se ajustan únicamente con train. Validación y test reciben esas transformaciones sin recalcular estadísticas \cite{sklearnleakage}.",
             r"Cada familia compara su configuración estándar y las variantes declaradas. Estándar significa parámetros predeterminados efectivos de la implementación utilizada, con las opciones de reproducibilidad del framework; no un valor atemporal independiente de la versión. Se registra \texttt{get\_params()} para comprobar qué se ejecutó. Los candidatos fallidos se registran con su error y sin puntuación.",
             r"Se elige una configuración por familia y un algoritmo global mediante la métrica de validación declarada en cada ejecución. Los empates se resuelven con el orden del catálogo. Los ganadores conservan el ajuste realizado en train: no se reajustan con validación. Sólo después de cerrar la selección se calculan sus predicciones y métricas de test. No se calcula desempeño de test de los candidatos descartados ni se usa test para elegir hiperparámetros."]
    filas = []
    for evidencia in evidencias:
        exp = evidencia.experimento
        filas.append([tex(evidencia.nombre.capitalize()), str(exp["semilla"]),
                      str(exp["muestra_train"]), str(exp["muestra_validacion"]), str(exp["muestra_test"]),
                      tex(exp["metrica_seleccion"])])
    texto.append(tabla(["Dataset", "Semilla", "Train", "Val.", "Test", "Selección"], filas,
                       caption="Protocolo efectivamente exportado por el framework.", alineacion="lrrrrl"))
    for evidencia in evidencias:
        prep = evidencia.experimento["preprocesamiento"]
        texto.append(tex(evidencia.nombre.capitalize()) + ": imputación " + ("activada" if prep["imputar"] else "desactivada")
                     + ", estandarización " + ("activada" if prep["estandarizar"] else "desactivada")
                     + ", inclusión de categóricas " + ("activada" if prep["incluir_categoricas"] else "desactivada")
                     + ". Las transformaciones descartadas o los atributos resultantes se conservan en los metadatos de cada ganador.")
    texto.append(r"Un único holdout proporciona una estimación dependiente de esta partición. Las comparaciones son descriptivas; no se ejecutaron repeticiones, validación externa ni una prueba de significación. Las métricas no representan eficacia clínica ni garantizan potabilidad real. El análisis estudia capacidad predictiva dentro de la muestra y no relaciones causales.")
    texto.append(r"\FloatBarrier")
    return "\n\n".join(texto)


def _comando(evidencia: Evidencia) -> str:
    config, args = evidencia.configuracion, evidencia.configuracion["argumentos"]
    lineas = ["python -m scripts.ejecutar_lab02 data/lab02/" + config["dataset"]["archivo"] + " \\",
              "  --target " + args["target"] + " \\",
              "  --salida salida_lab02/" + evidencia.nombre + " \\",
              "  --semilla " + str(args["semilla"]) + " \\",
              "  --test-size " + str(args["test_size"]) + " \\",
              "  --validation-size " + str(args["validation_size"])]
    extras = []
    if args.get("features"):
        extras.append("  --features " + (" \\" + "\n    ").join(args["features"]))
    for clave, flag in (("incluir_categoricas", "--incluir-categoricas"),
                       ("sin_escalar", "--sin-escalar"), ("sin_imputar", "--sin-imputar")):
        if args.get(clave):
            extras.append("  " + flag)
    for clave, defecto in (("separador", ","), ("decimal", "."), ("encoding", "utf-8")):
        if args.get(clave, defecto) != defecto:
            extras.append("  --" + clave + " " + repr(args[clave]))
    if args.get("algoritmos"):
        extras.append("  --algoritmos " + " ".join(args["algoritmos"]))
    if args.get("fuente"):
        extras.append("  --fuente " + args["fuente"])
    return " \\\n".join(["\n".join(lineas), *extras])


def _apendice(evidencias: list[Evidencia], procedencia: dict[str, Any]) -> str:
    texto = [r"\appendix\section{Reproducción, integridad y errores}",
             "Los siguientes comandos usan la sintaxis de continuación de una shell POSIX. En PowerShell se pueden introducir en una sola línea. Deben ejecutarse desde la raíz, con los CSV en las rutas indicadas y una carpeta de salida nueva. Las dependencias y sus versiones se registran por ejecución. La semilla conserva la partición, pero no garantiza resultados idénticos entre versiones, plataformas o implementaciones."]
    for evidencia in evidencias:
        texto.append(r"\subsection{" + tex(evidencia.nombre.capitalize()) + "}")
        if evidencia.nombre == "diabetes":
            origen = procedencia["diabetes"]
            texto.append(r"Antes del experimento se reproduce la preparación determinista del original. El script conserva sus bytes, deriva la clase documentada, excluye objetivos ausentes y exporta el mapa de filas; no imputa ni escala predictoras. Si ya existe una preparación idéntica puede repetirse; una entrada distinta requiere una carpeta limpia para no mezclar evidencia.")
            texto.append(r"\begin{verbatim}" + "\n"
                         + "python -m scripts.preparar_diabetes_lab02 \\\n  data/lab02/" + origen["original"]["ruta"] + " \\\n  --salida data/lab02\n"
                         + r"\end{verbatim}")
            texto.append(r"La regla y los hashes de los archivos se registran en \path{procedencia.json}. El mapa de filas se conserva en \path{trazabilidad_diabetes.csv}. Sus columnas relacionan \texttt{fila\_\allowbreak{}preparada}, \texttt{fila\_\allowbreak{}original} e \texttt{id\_\allowbreak{}original}. Los índices de las particiones experimentales corresponden al CSV preparado y se enlazan al original mediante esa tabla. No se mezcla el índice de fila con un identificador de paciente.")
        # URLs se presentan fuera del bloque para evitar líneas de código largas.
        comando = _comando(evidencia)
        comando = "\n".join(l for l in comando.splitlines() if "--fuente " not in l).rstrip(" \\\n")
        texto.append(r"\begin{verbatim}" + "\n" + comando + "\n" + r"\end{verbatim}")
        texto.append("Añadir --fuente con la URL registrada en la sección de procedencia conserva la atribución al repetir la exportación.")
        versiones = {**evidencia.configuracion.get("versiones", {}), **evidencia.experimento.get("versiones", {})}
        texto.append(pares([(paquete, tex(version)) for paquete, version in versiones.items()], "Versiones registradas durante el ajuste."))
        fallos = [f for f in evidencia.comparacion if f["estado"] != "ok"]
        errores_test = evidencia.experimento.get("errores_prueba", {})
        if not fallos and not errores_test:
            texto.append("No se registraron candidatos fallidos ni errores de evaluación final en esta ejecución.")
        else:
            texto.append("La ejecución conserva los siguientes errores; el informe no los reemplaza por resultados exitosos:")
            for f in fallos:
                texto.append(r"\begin{quote}\small " + tex(f["algoritmo"] + "/" + f["configuracion"] + ": " + f["error"]) + r"\end{quote}")
            for algoritmo, error in errores_test.items():
                texto.append(r"\begin{quote}\small " + tex(algoritmo + ": " + str(error)) + r"\end{quote}")
        texto.append("Los índices completos y los parámetros efectivos permanecen en los JSON; las predicciones y matrices se conservan por familia. Se verificó que los índices son únicos y disjuntos, que el score seleccionado maximiza validación y que accuracy, métricas por clase y F1 macro coinciden con las matrices de test.")
    texto.append(r"El archivo \texttt{manifest\_informe.json} registra SHA-256 de los archivos fuente usados para generar este documento. El proyecto de Overleaf contiene el texto, las tablas y las figuras embebidas; no necesita conexiones de red ni archivos de imagen para compilar.")
    return "\n\n".join(texto)


def _discusion(evidencias: list[Evidencia]) -> str:
    """Sintetiza comparaciones existentes sin seleccionar de nuevo con test."""
    texto = [r"\section{Discusión comparativa y limitaciones}",
             r"\subsection{Configuración estándar y variaciones}",
             "Las tablas por familia permiten comparar el estándar con sus variantes bajo los mismos índices y preprocesamiento. El criterio de elección corresponde a validación; la prueba final describe la generalización del ganador. Una variante ganadora no demuestra que cada cambio de hiperparámetro haya contribuido por separado, porque algunas configuraciones cambian varios valores simultáneamente."]
    for evidencia in evidencias:
        variantes, estandares = [], []
        for algoritmo in ALGORITMOS:
            if algoritmo not in evidencia.mejores:
                continue
            fila = fila_seleccionada(evidencia, algoritmo)
            (estandares if fila["configuracion"] == "estandar" else variantes).append(algoritmo)
        partes = []
        if variantes:
            partes.append("se eligen variantes en " + tex(", ".join(variantes)))
        if estandares:
            partes.append("se conserva el estándar en " + tex(", ".join(estandares)))
        texto.append("En " + tex(evidencia.nombre.capitalize()) + ", " + "; ".join(partes)
                     + ". La ausencia de mejora frente al estándar sólo se refiere al catálogo ensayado, no a todas las configuraciones posibles. Los empates documentados se resuelven por orden y no constituyen diferencias demostradas.")
    texto.append(r"\subsection{Métricas globales, balance y errores por clase}")
    for evidencia in evidencias:
        exp = evidencia.experimento
        seleccionado = exp["mejor_algoritmo_por_validacion"]
        if seleccionado not in evidencia.mejores:
            continue
        metricas = evidencia.mejores[seleccionado]["metricas_test"]
        mayor_accuracy = max(numero(r["metricas_test"]["accuracy"]) for r in evidencia.mejores.values())
        mayor_f1 = max(numero(r["metricas_test"]["f1"]["macro"]) for r in evidencia.mejores.values())
        top_accuracy = [a for a, r in evidencia.mejores.items()
                        if math.isclose(numero(r["metricas_test"]["accuracy"]), mayor_accuracy, abs_tol=1e-12)]
        top_f1 = [a for a, r in evidencia.mejores.items()
                  if math.isclose(numero(r["metricas_test"]["f1"]["macro"]), mayor_f1, abs_tol=1e-12)]
        texto.append("En " + tex(evidencia.nombre.capitalize()) + ", la mayor accuracy observada en test corresponde a "
                     + tex(", ".join(top_accuracy)) + " (" + porcentaje(mayor_accuracy)
                     + ") y el mayor F1 macro de test a " + tex(", ".join(top_f1)) + " ("
                     + puntuacion(mayor_f1) + "). Esta lectura descriptiva de las tablas no reemplaza la selección de "
                     + tex(seleccionado) + " realizada previamente con validación. Favorecer la exactitud global y favorecer el balance por clase son criterios distintos.")
        minoritaria = min(exp["distribuciones"]["train"], key=exp["distribuciones"]["train"].get)
        clase = metricas["precision"]["por_clase"][minoritaria]
        val_n = int(exp["distribuciones"]["validacion"].get(minoritaria, 0))
        test_n = int(exp["distribuciones"]["test"].get(minoritaria, 0))
        texto.append("La clase menos frecuente de train es " + tex(minoritaria)
                     + f"; tiene {val_n} observaciones en validación y {test_n} en test. En el modelo seleccionado, su recall de test es "
                     + porcentaje(clase["recall"]) + ", precisión " + porcentaje(clase["precision"])
                     + " y F1 " + puntuacion(clase["f1"])
                     + ". Los falsos negativos de esta clase y las falsas asignaciones a ella, ya contados en las matrices, muestran limitaciones que una sola cifra global oculta.")
    texto.extend((r"\subsection{Alcance de la comparación y limitaciones}",
                  "Las puntuaciones de Potabilidad y Diabetes no se promedian ni se utilizan para afirmar que una tarea sea intrínsecamente más fácil. Son muestras distintas, con etiquetas, atributos, proporciones de clase y preparación diferentes. El menor soporte de una clase produce estimaciones más sensibles a unos pocos aciertos o errores. La referencia de clase mayoritaria calculada para cada dataset ayuda a contextualizar la exactitud, pero no sustituye las métricas por clase.",
                  "El experimento utiliza un único holdout estratificado con la semilla registrada y un catálogo finito de variantes. No estima intervalos de confianza, estabilidad entre semillas ni significación estadística de diferencias pequeñas. La validación guía varias decisiones y puede favorecer configuraciones adaptadas a esa partición; el test separado conserva una evaluación posterior, pero no autoriza seguir ajustando tras inspeccionarlo. Los ganadores se mantienen ajustados con train, sin ampliar el entrenamiento con validación.",
                  "La imputación, el escalado y la codificación se aprendieron sólo con train, evitando usar estadísticas de validación o test durante el ajuste. En Diabetes se omitió glyhb de las predictoras porque la regla que define el objetivo depende directamente de esa medida. La exclusión de los objetivos ausentes limita el análisis a los casos con etiqueta derivable; no se acredita que esa ausencia sea aleatoria. Las fuentes y el esquema disponible tampoco prueban igualdad completa con los archivos adaptados del profesor ni representatividad externa. No se ejecutó evaluación clínica independiente. Los modelos son experimentos educativos: no certifican potabilidad, no proporcionan diagnóstico y no establecen causalidad.",
                  r"\FloatBarrier"))
    return "\n\n".join(texto)


def construir_documento(evidencias: list[Evidencia], autores: list[str], procedencia: dict[str, Any],
                       fecha: str, *, permitir_parcial: bool = False) -> str:
    recursos = Path(__file__).resolve().parents[1] / "docs" / "lab02"
    portada = r"\title{Lab 02: Métodos supervisados de clasificación}" + "\n"
    portada += r"\author{" + r"\\".join(tex(a) for a in (autores or ["Autores no especificados"])) + "}\n"
    portada += r"\date{EIF420O -- Universidad Nacional de Costa Rica\\" + tex(fecha) + "}\n"
    documento = [PREAMBULO, portada, r"\begin{document}\maketitle"]
    if not autores:
        documento.append(r"\noindent\fbox{\parbox{0.95\linewidth}{El documento no incluye datos de autoría.}}")
    if permitir_parcial:
        documento.append(r"\noindent\fbox{\parbox{0.95\linewidth}{Informe parcial: pueden faltar familias o variantes. No acredita por sí solo todos los requisitos de LAB 02.}}")
    documento.extend((r"\section{Introducción y objetivos}",
                      "El objetivo es comparar KNN, árboles de decisión, Random Forest, XGBoost y AdaBoost sobre los conjuntos de Potabilidad y Diabetes, incorporando configuración estándar y variaciones, selección de un modelo por familia y análisis de resultados. El trabajo integra los métodos en las clases supervisadas del framework Python del curso y conserva evidencia para repetir la experimentación.",
                      "Se documentan el origen de cada entrada y la preparación de su objetivo. Diabetes utiliza una regla de etiquetado respaldada por la documentación histórica del conjunto. El alcance de cada entrada se describe en su sección de datos y resultados. Las tablas y figuras se obtienen de los resultados experimentales exportados.",
                      (recursos / "fundamentos.tex").read_text(encoding="utf-8"),
                      _metodologia(evidencias)))
    conclusiones = [r"\section{Conclusiones}"]
    for evidencia in evidencias:
        documento.extend((r"\clearpage", _descripcion_datos(evidencia, procedencia)))
        for algoritmo in ALGORITMOS:
            if algoritmo in evidencia.mejores:
                documento.append(_resultado_familia(evidencia, algoritmo))
        ganador = evidencia.experimento["mejor_algoritmo_por_validacion"]
        if ganador in evidencia.mejores:
            seleccionado = fila_seleccionada(evidencia, ganador)
            metricas = evidencia.mejores[ganador]["metricas_test"]
            conclusiones.append("En " + tex(evidencia.nombre.capitalize()) + ", se selecciona " + tex(ganador)
                                + " con configuración " + tex(seleccionado["configuracion"])
                                + " por su score de validación " + puntuacion(seleccionado["puntuacion_validacion"])
                                + ". Su evaluación final obtiene accuracy " + porcentaje(metricas["accuracy"])
                                + " y F1 macro " + puntuacion(metricas["f1"]["macro"])
                                + ". Los resultados por clase y los conteos de errores muestran qué etiquetas el modelo reconoce y cuáles omite; no basta con la exactitud global para juzgar el balance.")
    conclusiones.append("Las configuraciones ganadoras dependen de la muestra, las predictoras y el holdout declarado. No se concluye superioridad universal, significación estadística ni causalidad. La selección se cerró antes de test y las transformaciones se aprendieron sólo en train. La evidencia conservada permite revisar parámetros, particiones y errores. Un trabajo posterior puede estudiar estabilidad mediante validación cruzada y una evaluación externa, fijando ese protocolo antes de utilizar nuevos datos de prueba.")
    documento.extend((r"\clearpage", _discusion(evidencias), r"\clearpage",
                      "\n\n".join(conclusiones), r"\FloatBarrier\clearpage"))
    referencias = (recursos / "referencias.tex").read_text(encoding="utf-8")
    fuentes = []
    for evidencia in evidencias:
        fuente = evidencia.configuracion["dataset"].get("fuente") or procedencia.get(evidencia.nombre, {}).get("fuente")
        if fuente:
            fuentes.append(r"\bibitem{datos" + evidencia.nombre + "}\nFuente del dataset de " + tex(evidencia.nombre.capitalize()) + ". " + url(fuente) + ".\n")
    documento.append(referencias.replace(r"\end{thebibliography}", "\n".join(fuentes) + r"\end{thebibliography}"))
    documento.extend((r"\clearpage", _apendice(evidencias, procedencia)))
    documento.append(r"\end{document}")
    return "\n\n".join(documento) + "\n"


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Informe LAB 02 desde resultados reales verificados; no entrena ni descarga.")
    parser.add_argument("resultados", type=Path, help="Contiene subcarpetas potabilidad y diabetes de la CLI.")
    parser.add_argument("--salida", type=Path, default=Path("output/lab02/informe"))
    parser.add_argument("--autor", action="append", default=[], help="Nombre completo (ID); repetir por autor.")
    parser.add_argument("--fecha", default="14 de octubre de 2026")
    parser.add_argument("--procedencia", type=Path, help="JSON por dataset con URL y SHA-256 de las entradas.")
    parser.add_argument("--permitir-parcial", action="store_true", help="Marca explícitamente un informe incompleto.")
    return parser


def ejecutar(args: argparse.Namespace) -> Path:
    evidencias = [cargar_evidencia(nombre, args.resultados / nombre, permitir_parcial=args.permitir_parcial)
                  for nombre in ("potabilidad", "diabetes")]
    procedencia = _json(args.procedencia) if args.procedencia else {}
    documento = construir_documento(evidencias, args.autor, procedencia, args.fecha,
                                    permitir_parcial=args.permitir_parcial)
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta = args.salida / "main.tex"
    ruta.write_text(documento, encoding="utf-8", newline="\n")
    manifest = {
        "autores_confirmados": args.autor,
        "fecha_documento": args.fecha,
        "parcial": args.permitir_parcial,
        "criterio": "métricas verificadas con matrices; selección sólo en validación",
        "fuentes": {e.nombre: {"dataset_sha256": e.configuracion["dataset"]["sha256"],
                               "archivos_resultados_sha256": e.hashes} for e in evidencias},
        "original_diabetes": procedencia.get("diabetes", {}).get("original"),
        "derivacion_diabetes": procedencia.get("diabetes", {}).get("derivacion"),
        "main_tex_sha256": _hash(ruta),
    }
    (args.salida / "manifest_informe.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    leeme = ("# Proyecto Overleaf de LAB 02\n\nImporte el ZIP en un proyecto vacío y compile main.tex con pdfLaTeX. "
             "Las tablas, figuras PGFPlots y referencias están embebidas; no se requieren imágenes, CSV externos ni ejecución de BibTeX.\n\n"
             "Los autores se especifican mediante --autor al ejecutar scripts.generar_informe_lab02.\n\n"
             "Las cifras proceden de las exportaciones experimentales, cuyos hashes constan en manifest_informe.json. "
             "El generador comprueba particiones, selección y coherencia de métricas y matrices. "
             "Este proyecto contiene las fuentes del informe; el código, los datos y los resultados se conservan en sus carpetas correspondientes.\n")
    (args.salida / "README.md").write_text(leeme, encoding="utf-8")
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
    print(f"Fuentes LaTeX y proyecto Overleaf guardados en {ruta.parent}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
