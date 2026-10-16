"""Invalidación de evidencia de regresión independiente de los widgets."""

from .configuracion_regresion import firma_regresion
from .estado_clasificacion import firma_parametros, huella_datos


def limpiar_resultados_regresion(estado) -> None:
    """Limpia snapshots, intentos y errores sin alterar controles ni particiones."""
    for clave in tuple(estado):
        if clave.startswith(
            ("resultado_lab3_", "contexto_lab3_", "error_lab3_", "lab3_firma_")
        ):
            estado.pop(clave, None)
    estado.pop("regresion_contexto", None)


def sincronizar_contexto_regresion(estado, datos, configuracion: dict) -> tuple:
    """Las ediciones de parámetros no invalidan una ejecución previa exitosa."""
    firma = (
        estado.get("dataset_identidad"),
        huella_datos(datos),
        firma_parametros(estado.get("dataset_preparacion", {})),
        *firma_regresion(configuracion),
    )
    if estado.get("regresion_contexto") != firma:
        limpiar_resultados_regresion(estado)
        estado["regresion_contexto"] = firma
    return firma
