"""Genera un informe LaTeX autónomo desde evidencia de regresión verificada."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

from framework_ia.datos.concrete_lab03 import (
    COLUMNAS_CONCRETE,
    PREDICTORAS_CONCRETE,
    validar_preparacion_concrete,
)
from framework_ia.evidencia import cargar_evidencia

RAIZ = Path(__file__).resolve().parents[1]
RECURSOS = RAIZ / "docs" / "lab03"
NOMBRES = {
    "RLS": "Lineal simple",
    "RLM": "Lineal múltiple",
    "Lasso": "Lasso",
    "LassoCV": "LassoCV",
    "Ridge": "Ridge",
    "RidgeCV": "RidgeCV",
    "SVR": "SVR",
    "DT": "Árbol",
    "RF": "Random Forest",
}
FAMILIAS = {
    "Lineales": ("RLS", "RLM", "Lasso", "LassoCV", "Ridge", "RidgeCV"),
    "Vectores de soporte": ("SVR",),
    "Basados en árboles": ("DT", "RF"),
}
ETIQUETAS = {
    "cement": "Cemento",
    "slag": "Escoria",
    "fly_ash": "Ceniza volante",
    "water": "Agua",
    "superplasticizer": "Superplastificante",
    "coarse_aggregate": "Agregado grueso",
    "fine_aggregate": "Agregado fino",
    "age": "Edad",
    "strength": "Resistencia",
}


def tex(valor) -> str:
    """Escapa valores de datos y argumentos; los recursos son LaTeX controlado."""
    mapa = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapa.get(c, c) for c in str(valor)).replace("\n", " ")


def numero(valor, decimales=3) -> str:
    valor = float(valor)
    if not math.isfinite(valor):
        raise ValueError("El informe no admite valores no finitos.")
    return f"{valor:.{decimales}f}"


def tabla(encabezados, filas, columnas, titulo) -> str:
    contenido = [
        r"\begin{table}[H]\centering\small",
        r"\caption{" + titulo + "}",
        r"\begin{tabular}{" + columnas + r"}\toprule",
        " & ".join(encabezados) + r" \\\midrule",
    ]
    contenido.extend(" & ".join(fila) + r" \\" for fila in filas)
    contenido.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(contenido)


def coordenadas(x, y) -> str:
    return " ".join(
        f"({float(a):.7g},{float(b):.7g})" for a, b in zip(x, y, strict=True)
    )


def figura(coordenadas_xy, opciones, titulo, lineas="") -> str:
    return (
        r"\begin{figure}[H]\centering" + "\n"
        r"\begin{tikzpicture}\begin{axis}[width=.92\linewidth,height=6cm,"
        r"grid=major,grid style={gray!15},tick label style={font=\small},"
        + opciones
        + "]\n"
        + lineas
        + "\n"
        r"\addplot[only marks,mark=*,mark size=1.2pt,blue!65!black,opacity=.65] coordinates {"
        + coordenadas_xy
        + "};\n"
        r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{" + titulo + r"}\end{figure}"
    )


def construir_informe(evidencia, autores: list[str], fecha: str) -> str:
    """Construye texto y gráficos con los números de la ejecución, sin entrenar."""
    date.fromisoformat(fecha)
    if not autores or any(not str(a).strip() for a in autores):
        raise ValueError("Debe indicar autores no vacíos.")
    if len(set(autores)) != len(autores):
        raise ValueError("Los autores no deben repetirse.")
    e = evidencia
    meta = e.configuracion["experimento"]
    datos = e.datos
    if (
        e.procedencia.get("dataset") != "uci_concrete_compressive_strength"
        or list(datos.columns) != list(COLUMNAS_CONCRETE)
        or meta["target"] != "strength"
        or meta["features"] != list(PREDICTORAS_CONCRETE)
        or meta["feature_simple"] != "cement"
    ):
        raise ValueError(
            "El informe requiere Concrete con sus ocho predictoras, strength y RLS cement."
        )
    target = meta["target"]
    seleccionadas = e.comparacion[e.comparacion.seleccionado].copy()
    por_algoritmo = {r.algoritmo: r for r in seleccionadas.itertuples()}
    orden = list(NOMBRES)
    ranking = sorted(orden, key=lambda a: por_algoritmo[a].rmse_validacion)
    ganador = meta["mejor_algoritmo_por_validacion"]
    mejor_test = min(orden, key=lambda a: e.mejores[a]["metricas_test"]["rmse"])
    pred = e.predicciones[ganador]["test"]
    rmse_cv = por_algoritmo[ganador].rmse_validacion
    rmse_test = e.mejores[ganador]["metricas_test"]["rmse"]
    derivacion = e.procedencia.get("derivacion", {})
    filas_originales = derivacion.get("filas_originales", len(datos))
    duplicados = derivacion.get(
        "filas_eliminadas_duplicados", filas_originales - len(datos)
    )
    train = datos.loc[meta["train_indices"]]
    secciones = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[spanish,es-nodecimaldot,es-noshorthands]{babel}",
        r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern,microtype,booktabs,tabularx,amsmath,float,xurl,placeins}",
        r"\usepackage{pgfplots}\usetikzlibrary{arrows.meta,positioning,shapes.geometric}",
        r"\usepackage[font=small,labelfont=bf]{caption}\usepackage[hidelinks]{hyperref}",
        r"\usepackage[a4paper,margin=2.3cm]{geometry}\pgfplotsset{compat=1.18}",
        r"\setlength{\emergencystretch}{2em}\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\parskip}{.4em}\setlength{\parindent}{0pt}",
        (
            r"\title{\textbf{Laboratorio 3}\\Métodos supervisados: regresión y benchmarking\\"
            r"\large Predicción de la resistencia a compresión del concreto}"
        ),
        r"\author{" + r"\\".join(tex(a) for a in autores) + "}",
        r"\date{EIF420O -- Inteligencia Artificial\\Universidad Nacional de Costa Rica\\"
        r"Profesor: Dr. Juan Murillo-Morera\\" + tex(fecha) + "}",
        r"\begin{document}\maketitle",
        r"\begin{abstract}",
        (
            f"Se compararon nueve modelos de regresión y catorce configuraciones sobre {len(datos)} "
            "observaciones de Concrete Compressive Strength. La selección utilizó validación cruzada "
            "de cinco pliegues y un conjunto de prueba reservado del 25\\,\\%. "
            f"{tex(NOMBRES[ganador])} obtuvo el menor RMSE promedio de validación "
            f"({numero(rmse_cv)} MPa); su RMSE de prueba fue {numero(rmse_test)} MPa. "
            "Se examinan regularización, complejidad, diferencias entre familias y límites de generalización."
        ),
        r"\end{abstract}",
        r"\section{Introducción y objetivos}",
        (
            "El objetivo consiste en predecir una cantidad continua, la resistencia a compresión, "
            "a partir de los componentes de una mezcla y su edad. Este caso permite contrastar "
            "representaciones lineales, basadas en kernels y basadas en árboles bajo un protocolo común. "
            "Se busca comparar modelos dentro de una misma familia y entre familias, y justificar "
            "la selección mediante evidencia de validación, sin utilizar la prueba para elegir hiperparámetros."
        ),
        r"\section{Datos y análisis exploratorio}",
        (
            "El conjunto de I.-C. Yeh está publicado por UCI, con licencia CC BY 4.0. "
            "Su ficha describe 1.030 observaciones, ocho predictoras y una respuesta numérica, "
            "sin valores faltantes. El archivo original contiene mediciones de laboratorio y datos "
            r"sin escalar \cite{uci_concrete,yeh1998}."
        ),
        (
            f"La preparación verificó {filas_originales} filas originales y retiró {duplicados} "
            f"repeticiones completas, conservando {len(datos)} filas. No se imputaron ni escalaron "
            "los datos durante la conversión. Se conservaron las observaciones con las mismas "
            "predictoras y distinta respuesta, y no se eliminaron valores extremos."
        ),
    ]
    filas = []
    for nombre in datos:
        valores = datos[nombre]
        unidad = (
            "días" if nombre == "age" else ("MPa" if nombre == target else r"kg/m$^3$")
        )
        filas.append(
            [
                tex(ETIQUETAS.get(nombre, nombre)),
                unidad,
                numero(valores.min()),
                numero(valores.mean()),
                numero(valores.std()),
                numero(valores.max()),
            ]
        )
    secciones.append(
        tabla(
            ["Variable", "Unidad", "Mínimo", "Media", "DE", "Máximo"],
            filas,
            "llrrrr",
            "Resumen descriptivo del CSV preparado.",
        )
    )
    correlaciones = train.corr(numeric_only=True)[target].drop(target)
    secciones.append(
        tabla(
            ["Predictora", r"$r$ de Pearson"],
            [[tex(ETIQUETAS.get(c, c)), numero(v)] for c, v in correlaciones.items()],
            "lr",
            "Correlación con la respuesta, calculada sólo en entrenamiento.",
        )
    )
    mayor = correlaciones.abs().idxmax()
    secciones.append(
        f"En entrenamiento, {tex(ETIQUETAS.get(mayor, mayor)).lower()} presenta la mayor "
        f"correlación lineal absoluta con la respuesta ({numero(correlaciones[mayor])}). "
        "Estas asociaciones son descriptivas y no demuestran causalidad. Cemento se fijó "
        "como predictora del modelo simple antes del benchmark; la correlación no seleccionó "
        "variables ni modificó el conjunto de prueba. Los modelos restantes utilizan las ocho predictoras."
    )
    conteos, bordes = np.histogram(datos[target], bins=12)
    centros = (bordes[:-1] + bordes[1:]) / 2
    secciones.append(
        r"\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}["
        r"width=.92\linewidth,height=5cm,ybar,bar width=15pt,xlabel={Resistencia (MPa)},"
        r"ylabel={Frecuencia},ymin=0,grid=major,grid style={gray!15}]"
        r"\addplot[fill=blue!35,draw=blue!65!black] coordinates {"
        + coordenadas(centros, conteos)
        + r"};\end{axis}\end{tikzpicture}"
        r"\caption{Distribución descriptiva de la resistencia en los datos preparados.}\end{figure}"
    )
    secciones.extend(
        [
            (RECURSOS / "fundamentos.tex").read_text(encoding="utf-8"),
            r"\section{Metodología reproducible}",
        ]
    )
    secciones.append(
        "Los estimadores se ejecutaron mediante la jerarquía Supervisado--Regresion del framework "
        "del curso. La partición aleatoria, sin estratificación, utilizó semilla 42: "
        f"{len(meta['train_indices'])} filas para entrenamiento y {len(meta['test_indices'])} "
        "para prueba. Sobre entrenamiento se aplicó KFold de cinco pliegues mezclados con la "
        "misma semilla. Todos los candidatos compartieron filas y pliegues. "
        "El preprocesador se ajustó en cada entrenamiento externo; imputación por mediana "
        "y estandarización no utilizaron datos de validación o prueba. En este conjunto no "
        "hubo valores faltantes que imputar. No se aplicó selección automática de atributos."
    )
    secciones.append(
        "Se eligió una configuración por algoritmo mediante el menor RMSE promedio de validación. "
        "El mismo criterio determinó el ganador global; los empates conservaron el orden del catálogo. "
        "Se informa la desviación estándar muestral entre pliegues. Después de cerrar la selección, "
        "cada configuración elegida se reajustó con todo entrenamiento y se evaluó sobre prueba. "
        r"Esta separación evita elegir retrospectivamente por el resultado de prueba \cite{cawley2010}."
    )
    filas = []
    for algoritmo, candidatos in meta["configuraciones"].items():
        for candidato in candidatos:
            parametros = candidato["parametros"]
            descripcion = (
                ", ".join(f"{k}={v}" for k, v in parametros.items()) or "Estándar"
            )
            filas.append(
                [tex(NOMBRES[algoritmo]), tex(candidato["nombre"]), tex(descripcion)]
            )
    secciones.append(
        tabla(
            ["Modelo", "Configuración", "Parámetros"],
            filas,
            "llp{7.1cm}",
            "Catálogo de catorce candidatos.",
        )
    )
    versiones = e.configuracion["versiones"]
    secciones.append(
        tabla(
            ["Componente", "Versión"],
            [[tex(k), tex(v)] for k, v in versiones.items()],
            "ll",
            "Versiones registradas durante el entrenamiento.",
        )
    )
    secciones.extend(
        [
            r"\section{Resultados y comparación de modelos}",
            r"\subsection{Comparación de las configuraciones}",
        ]
    )
    filas = []
    for r in e.comparacion.itertuples():
        filas.append(
            [
                tex(r.algoritmo),
                tex(r.configuracion),
                numero(r.rmse_validacion),
                numero(r.puntuacion_validacion_std),
                "Sí" if r.seleccionado else "No",
            ]
        )
    secciones.append(
        tabla(
            ["Modelo", "Configuración", "RMSE CV", "DE", "Elegida"],
            filas,
            "llrrc",
            "Validación de todos los candidatos; errores en MPa.",
        )
    )
    secciones.append(r"\subsection{Comparación dentro de las familias}")
    base_simple = por_algoritmo["RLS"].rmse_validacion
    mejora = (
        100 * (base_simple - por_algoritmo["RLM"].rmse_validacion) / base_simple
        if base_simple > 0
        else 0
    )
    cambio = "reducción" if mejora >= 0 else "aumento"
    secciones.append(
        f"La regresión simple obtuvo RMSE de validación {numero(por_algoritmo['RLS'].rmse_validacion)} "
        f"MPa y la múltiple {numero(por_algoritmo['RLM'].rmse_validacion)} MPa: "
        f"la múltiple presentó una {cambio} relativa de {numero(abs(mejora), 1)}\\,\\% "
        "en este error. Esta comparación modifica también la información disponible: "
        "la simple utiliza cemento y la múltiple incorpora las ocho predictoras."
    )
    lineal = min(FAMILIAS["Lineales"], key=lambda a: por_algoritmo[a].rmse_validacion)
    secciones.append(
        f"Dentro del grupo lineal, {tex(NOMBRES[lineal])} alcanzó el menor RMSE promedio "
        f"({numero(por_algoritmo[lineal].rmse_validacion)} MPa). "
        f"LassoCV obtuvo {numero(por_algoritmo['LassoCV'].rmse_validacion)} MPa y RidgeCV "
        f"{numero(por_algoritmo['RidgeCV'].rmse_validacion)} MPa. La selección automática "
        "de alpha no asegura superar las configuraciones fijas en la validación externa. "
        "Las diferencias deben interpretarse junto con la variación entre pliegues."
    )
    secciones.append(
        tabla(
            ["Estimador", r"$\alpha$ del ajuste final", "Criterio interno"],
            [
                [
                    tex(a),
                    numero(e.mejores[a]["metadatos"]["alpha_elegido"], 4),
                    "MSE, cinco pliegues",
                ]
                for a in ("LassoCV", "RidgeCV")
            ],
            "lrl",
            "Regularización elegida internamente al reajustar sobre entrenamiento completo.",
        )
    )
    arboles = min(
        FAMILIAS["Basados en árboles"], key=lambda a: por_algoritmo[a].rmse_validacion
    )
    secciones.append(
        f"Entre los modelos basados en árboles, {tex(NOMBRES[arboles])} fue el mejor validado. "
        f"El árbol individual registró {numero(por_algoritmo['DT'].rmse_validacion)} MPa y "
        f"Random Forest {numero(por_algoritmo['RF'].rmse_validacion)} MPa. La tabla de candidatos "
        "permite observar el efecto de limitar la profundidad a cinco niveles frente a permitir "
        "profundidad libre. SVR compara dos valores de C con kernel RBF; no se compararon kernels distintos."
    )
    secciones.append(r"\subsection{Comparación entre familias}")
    filas = []
    for familia, algoritmos in FAMILIAS.items():
        mejor = min(algoritmos, key=lambda a: por_algoritmo[a].rmse_validacion)
        filas.append(
            [
                tex(familia),
                tex(NOMBRES[mejor]),
                numero(por_algoritmo[mejor].rmse_validacion),
            ]
        )
    secciones.append(
        tabla(
            ["Familia", "Mejor modelo por validación", "RMSE CV"],
            filas,
            "llr",
            "Mejor configuración de cada familia conceptual.",
        )
    )
    secciones.append(
        f"El ganador global fue {tex(NOMBRES[ganador])}, configuración "
        f"{tex(por_algoritmo[ganador].configuracion)}, con {numero(rmse_cv)} MPa de RMSE "
        f"promedio y DE {numero(por_algoritmo[ganador].puntuacion_validacion_std)} MPa. "
        f"La siguiente alternativa fue {tex(NOMBRES[ranking[1]])}, con "
        f"{numero(por_algoritmo[ranking[1]].rmse_validacion)} MPa. "
        "La diferencia observada no constituye una prueba de significancia estadística ni "
        "demuestra superioridad sobre todos los datasets o particiones."
    )
    secciones.append(
        f"El RMSE de entrenamiento del árbol fue {numero(e.mejores['DT']['metricas_train']['rmse'])} "
        f"MPa, frente a {numero(por_algoritmo['DT'].rmse_validacion)} MPa en validación. "
        f"Random Forest registró {numero(e.mejores['RF']['metricas_train']['rmse'])} MPa en "
        f"entrenamiento y {numero(por_algoritmo['RF'].rmse_validacion)} MPa en validación. "
        "La mayor capacidad de ajustar entrenamiento del árbol individual no se tradujo "
        "en menor error externo. El bosque obtuvo mejor validación aun con un ajuste "
        "menos exacto en entrenamiento, comportamiento compatible con la reducción de "
        r"variabilidad mediante el ensamble \cite{breiman2001}."
    )
    secciones.append(r"\subsection{Evaluación final en prueba}")
    filas = []
    for a in orden:
        m = e.mejores[a]
        filas.append(
            [
                tex(a),
                numero(m["metricas_train"]["rmse"]),
                numero(m["metricas_test"]["rmse"]),
                numero(m["metricas_test"]["mae"]),
                numero(m["metricas_test"]["mse"]),
                numero(m["metricas_test"]["r2"]),
            ]
        )
    secciones.append(
        tabla(
            [
                "Modelo",
                "RMSE train",
                "RMSE test",
                "MAE test",
                "MSE test",
                r"$R^2$ test",
            ],
            filas,
            "lrrrrr",
            "Evaluación de las configuraciones seleccionadas por validación.",
        )
    )
    secciones.append(
        f"{tex(NOMBRES[mejor_test])} obtuvo el menor RMSE de prueba entre las configuraciones "
        f"evaluadas ({numero(e.mejores[mejor_test]['metricas_test']['rmse'])} MPa). "
        f"El modelo seleccionado sigue siendo {tex(NOMBRES[ganador])}, porque la decisión "
        "se cerró utilizando exclusivamente validación. Los errores están expresados en MPa, "
        r"salvo MSE, cuya unidad es MPa$^2$; $R^2$ es adimensional."
    )
    limite_inf = min(pred.real.min(), pred.prediccion.min())
    limite_sup = max(pred.real.max(), pred.prediccion.max())
    secciones.append(
        figura(
            coordenadas(pred.real, pred.prediccion),
            "xlabel={Valor real (MPa)},ylabel={Predicción (MPa)}",
            "Predicciones de prueba del ganador por validación. La diagonal representa predicción perfecta.",
            r"\addplot[black,dashed,domain="
            + f"{limite_inf:.5g}:{limite_sup:.5g}"
            + r"] {x};",
        )
    )
    secciones.append(
        figura(
            coordenadas(pred.prediccion, pred.residuo),
            "xlabel={Predicción (MPa)},ylabel={Residuo (MPa)}",
            "Residuos de prueba del ganador; residuo = real menos predicción.",
            r"\addplot[black,dashed,domain="
            + f"{pred.prediccion.min():.5g}:{pred.prediccion.max():.5g}"
            + r"] {0};",
        )
    )
    secciones.append(
        f"El residuo medio de prueba fue {numero(pred.residuo.mean())} MPa y el mayor error "
        f"absoluto fue {numero(pred.residuo.abs().max())} MPa. "
        "Un residuo positivo indica subestimación y uno negativo indica sobreestimación. "
        "Los gráficos permiten inspeccionar dispersión y sesgos locales; el residuo medio "
        "cercano a cero no garantiza errores pequeños en cada observación."
    )
    secciones.extend(
        [
            r"\section{Discusión y limitaciones}",
            (
                "La deduplicación evita repartir réplicas completas entre conjuntos, pero no asegura "
                "independencia entre mezclas relacionadas o mediciones a edades distintas. La evaluación "
                "por filas representa observaciones de una distribución semejante a la disponible; "
                "no demuestra generalización a formulaciones, lotes o laboratorios nuevos. "
                "Sólo se utilizaron una semilla, un dataset y el catálogo declarado. No se evaluaron "
                "todos los hiperparámetros posibles ni se realizaron pruebas de significancia."
            ),
            (
                "En LassoCV y RidgeCV, el preprocesamiento se aprende con el entrenamiento externo "
                "antes de la validación interna de alpha. No se reajusta en cada pliegue interno; "
                "sus puntuaciones internas no se presentan como evidencia independiente. El benchmark "
                r"sí reajusta las transformaciones en cada pliegue externo \cite{sklearn_fuga,cawley2010}."
            ),
            r"\section{Conclusiones}",
            (
                f"El protocolo comparó nueve modelos y catorce configuraciones bajo particiones comunes. "
                f"La elección fue {tex(NOMBRES[ganador])}, con RMSE de validación {numero(rmse_cv)} MPa "
                f"y RMSE de prueba {numero(rmse_test)} MPa. La comparación simple--múltiple examinó "
                "el efecto de incorporar información adicional; Lasso/Ridge y sus versiones CV "
                "permitieron estudiar regularización; árbol/Random Forest mostraron el efecto de "
                "profundidad y ensamble. Las conclusiones se limitan a este dataset, partición y "
                "configuraciones, con la incertidumbre descrita entre pliegues."
            ),
            r"\section{Reproducción y trazabilidad}",
            (
                "La preparación conserva el XLS original y registra los hashes de los bytes originales "
                "y del CSV materializado. La evidencia exportada conserva predicciones, residuos, "
                "métricas, parámetros e índices. La generación del informe valida su coherencia "
                "sin volver a entrenar. Los siguientes comandos reproducen la preparación y el experimento:"
            ),
            r"\begin{verbatim}",
            "from scripts.preparar_concrete_lab03 import main as preparar",
            "from scripts.ejecutar_lab03 import main as experimentar",
            "preparar(['data/lab03/concrete/Concrete_Data.xls'])",
            "experimentar([",
            "    'data/lab03/concrete/concrete.csv',",
            "    '--target', 'strength', '--feature-simple', 'cement',",
            "    '--salida', 'salida_lab03/concrete',",
            "    '--semilla', '42', '--test-size', '0.25',",
            "    '--estrategia', 'cv', '--folds', '5',",
            "    '--fuente',",
            "    'https://archive.ics.uci.edu/dataset/165/'",
            "    'concrete+compressive+strength',",
            "])",
            r"\end{verbatim}",
            (
                "El bloque Python se ejecuta desde la raíz del proyecto. Se requiere una "
                "carpeta de resultados vacía. El archivo de dependencias fijadas permite reproducir el "
                "entorno; otras versiones pueden cambiar resultados numéricos."
            ),
            r"\noindent SHA-256 del CSV: \par\noindent\texttt{"
            + tex(e.configuracion["dataset"]["sha256"])
            + "}.",
            (RECURSOS / "referencias.tex").read_text(encoding="utf-8"),
            r"\end{document}",
        ]
    )
    return "\n\n".join(secciones) + "\n"


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resultados", type=Path)
    parser.add_argument("--datos", type=Path, required=True)
    parser.add_argument("--procedencia", type=Path, required=True)
    parser.add_argument("--autor", action="append", required=True)
    parser.add_argument(
        "--fecha", required=True, help="Fecha ISO explícita: AAAA-MM-DD."
    )
    parser.add_argument(
        "--salida", type=Path, default=RAIZ / "output" / "lab03" / "informe"
    )
    return parser


def ejecutar(argumentos: argparse.Namespace) -> Path:
    evidencia = cargar_evidencia(
        argumentos.resultados, argumentos.datos, argumentos.procedencia
    )
    validar_preparacion_concrete(argumentos.procedencia.parent, evidencia.procedencia)
    contenido = construir_informe(evidencia, argumentos.autor, argumentos.fecha).encode(
        "utf-8"
    )
    manifest = {
        "informe_sha256": hashlib.sha256(contenido).hexdigest(),
        "dataset_sha256": evidencia.configuracion["dataset"]["sha256"],
        "evidencia": evidencia.hashes,
        "autores": argumentos.autor,
        "fecha": argumentos.fecha,
        "algoritmo_seleccionado": evidencia.configuracion["experimento"][
            "mejor_algoritmo_por_validacion"
        ],
    }
    archivos = {
        "main.tex": contenido,
        "manifest_informe.json": (
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    for nombre, esperado in archivos.items():
        ruta = argumentos.salida / nombre
        if ruta.exists() and (not ruta.is_file() or ruta.read_bytes() != esperado):
            raise ValueError(f"Ya existe un archivo distinto: {ruta}.")
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    for nombre, esperado in archivos.items():
        ruta = argumentos.salida / nombre
        if not ruta.exists():
            ruta.write_bytes(esperado)
    return argumentos.salida


def main(argv: list[str] | None = None) -> int:
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        salida = ejecutar(argumentos)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"No se pudo generar el informe Lab03: {exc}\n")
    print(f"Informe LaTeX y manifiesto verificados en {salida}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
