
"""
Consolidación corregida de métricas V5-V9D.
No entrena modelos ni modifica archivos existentes.
"""

from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

ARCHIVOS = [
    "metricas_modelo_v5.json",
    "metricas_modelo_v6.json",
    "metricas_modelo_v7.json",
    "metricas_modelo_v8.json",
    "metricas_modelo_v8b.json",
    "metricas_modelo_v9b.json",
    "metricas_modelo_v9c_preventivo.json",
    "metricas_modelo_v9d_preventivo.json",
    "comparacion_diseno_modelos.json",
]

SALIDA_CSV = MODELS / "comparativa_modelos_consolidada.csv"
SALIDA_TXT = MODELS / "resumen_modelos_consolidado.txt"


def primero(*valores):
    for valor in valores:
        if valor is not None:
            return valor
    return None


def extraer_reporte(datos, nombre_clase="Con Muertos"):
    reporte = (
        datos.get("reporte_clasificacion")
        or datos.get("classification_report")
        or datos.get("reporte_por_clase")
        or {}
    )

    # El reporte puede venir dentro de un bloque de prueba.
    if not reporte:
        reporte = datos.get("prueba", {}).get(
            "reporte_clasificacion", {}
        )

    return reporte.get(nombre_clase, {})


def obtener_metricas(datos, bloque=None):
    """
    Normaliza métricas con distintos nombres y estructuras.
    Para V8B utiliza el bloque 'prueba'.
    """
    fuente = datos

    if bloque and isinstance(datos.get(bloque), dict):
        fuente = datos[bloque]

    muertos = fuente.get("con_muertos", {})
    if not muertos:
        muertos = datos.get("con_muertos", {})

    if not muertos and isinstance(datos.get("prueba"), dict):
        muertos = datos["prueba"].get("con_muertos", {})

    reporte_muertos = extraer_reporte(fuente)
    if not reporte_muertos:
        reporte_muertos = extraer_reporte(datos)

    accuracy = primero(
        fuente.get("accuracy"),
        datos.get("accuracy"),
    )

    balanced = primero(
        fuente.get("balanced_accuracy"),
        datos.get("balanced_accuracy"),
    )

    macro_f1 = primero(
        fuente.get("f1_macro"),
        fuente.get("macro_f1"),
        datos.get("f1_macro"),
        datos.get("macro_f1"),
        fuente.get("macro avg", {}).get("f1-score"),
    )

    weighted_f1 = primero(
        fuente.get("f1_weighted"),
        fuente.get("weighted_f1"),
        datos.get("f1_weighted"),
        datos.get("weighted_f1"),
        fuente.get("weighted avg", {}).get("f1-score"),
    )

    precision_muertos = primero(
        fuente.get("precision_con_muertos"),
        fuente.get("precision_muertos"),
        muertos.get("precision"),
        reporte_muertos.get("precision"),
    )

    recall_muertos = primero(
        fuente.get("recall_con_muertos"),
        fuente.get("recall_muertos"),
        muertos.get("recall"),
        reporte_muertos.get("recall"),
    )

    f1_muertos = primero(
        fuente.get("f1_con_muertos"),
        fuente.get("f1_muertos"),
        muertos.get("f1"),
        reporte_muertos.get("f1-score"),
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "precision_muertos": precision_muertos,
        "recall_muertos": recall_muertos,
        "f1_muertos": f1_muertos,
    }


def tipo_modelo(nombre, datos):
    texto = (
        nombre + " " + str(datos.get("version", "")) + " "
        + str(datos.get("modelo", ""))
    ).lower()

    if "preventivo" in texto or "v9c" in texto or "v9d" in texto:
        return "Preventivo candidato"

    if "v9b" in texto:
        return "Temporal, incluye variables posteriores"

    return "Experimental; revisar variables y evaluación"


def fila_modelo(nombre, version, datos, bloque=None, variables=None):
    metricas = obtener_metricas(datos, bloque)

    if variables is None:
        variables = (
            datos.get("variables")
            or datos.get("variables_utilizadas")
            or datos.get("variables_predictoras")
            or []
        )

    train = primero(
        datos.get("train_rows"),
        datos.get("registros_entrenamiento"),
        datos.get("entrenamiento"),
    )
    test = primero(
        datos.get("test_rows"),
        datos.get("registros_prueba"),
    )

    if bloque == "prueba" and isinstance(datos.get("prueba"), dict):
        test = primero(
            datos["prueba"].get("registros_prueba"),
            test,
        )

    return {
        "version": version,
        "archivo": nombre,
        "tipo": tipo_modelo(nombre, datos),
        "bloque_metricas": bloque or "principal",
        "registros_entrenamiento": train,
        "registros_prueba": test,
        "accuracy": metricas["accuracy"],
        "balanced_accuracy": metricas["balanced_accuracy"],
        "f1_macro": metricas["f1_macro"],
        "f1_weighted": metricas["f1_weighted"],
        "precision_muertos": metricas["precision_muertos"],
        "recall_muertos": metricas["recall_muertos"],
        "f1_muertos": metricas["f1_muertos"],
        "variables": (
    " | ".join(map(str, variables))
    if isinstance(variables, (list, tuple, set))
    else str(variables) if variables is not None
    else ""
),
    }


def main():
    if not MODELS.exists():
        raise FileNotFoundError(f"No existe la carpeta: {MODELS}")

    filas = []
    errores = []

    for nombre in ARCHIVOS:
        ruta = MODELS / nombre

        if not ruta.exists():
            errores.append(f"No encontrado: {nombre}")
            continue

        try:
            with ruta.open("r", encoding="utf-8-sig") as archivo:
                datos = json.load(archivo)
        except (OSError, json.JSONDecodeError) as exc:
            errores.append(f"{nombre}: {exc}")
            continue

        if nombre == "comparacion_diseno_modelos.json":
            resultados = datos.get("resultados", {})

            for nombre_modelo, valores in resultados.items():
                if not isinstance(valores, dict):
                    continue

                metricas = {
                    "accuracy": valores.get("accuracy"),
                    "balanced_accuracy": valores.get("balanced_accuracy"),
                    "f1_macro": valores.get("f1_macro"),
                    "f1_weighted": valores.get("f1_weighted"),
                    "precision_muertos": valores.get("precision_muertos"),
                    "recall_muertos": valores.get("recall_muertos"),
                    "f1_muertos": valores.get("f1_muertos"),
                }

                filas.append({
                    "version": nombre_modelo,
                    "archivo": nombre,
                    "tipo": "Comparación experimental",
                    "bloque_metricas": "prueba 2019-2020",
                    "registros_entrenamiento": datos.get(
                        "registros_entrenamiento"
                    ),
                    "registros_prueba": datos.get("registros_prueba"),
                    **metricas,
                    "variables": " | ".join(
                        datos.get(
                            "variables_base" if nombre_modelo == "A_base"
                            else "variables_con_diseno",
                            [],
                        )
                    ),
                })
            continue

        if nombre == "metricas_modelo_v8b.json":
            filas.append(
                fila_modelo(nombre, "V8B", datos, bloque="prueba")
            )
            filas.append(
                fila_modelo(nombre, "V8B", datos, bloque="validacion")
            )
        else:
            version = (
                datos.get("version")
                or datos.get("modelo")
                or Path(nombre).stem
            )
            filas.append(
                fila_modelo(nombre, version, datos)
            )

    if not filas:
        raise ValueError("No se pudieron extraer métricas de los JSON.")

    tabla = pd.DataFrame(filas)

    # Orden legible: primero versiones antiguas, luego preventivas
    # y finalmente los experimentos de comparación.
    orden = {
        "V5": 5,
        "Random Forest V5": 5,
        "Random Forest V6": 6,
        "Random Forest V7": 7,
        "Random Forest V8": 8,
        "V8B": 8.5,
        "V9B": 9,
        "V9C_preventivo": 10,
        "V9D_preventivo": 11,
        "A_base": 12,
        "B_con_diseno": 13,
    }

    tabla["_orden"] = tabla["version"].map(orden).fillna(99)
    tabla = tabla.sort_values(
        ["_orden", "bloque_metricas"]
    ).drop(columns="_orden")

    tabla.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")

    columnas_vista = [
        "version",
        "bloque_metricas",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "precision_muertos",
        "recall_muertos",
        "f1_muertos",
    ]

    texto = [
        "CONSOLIDADO DE MODELOS",
        "=" * 75,
        "Las celdas vacías indican métricas no disponibles en el JSON.",
        "",
        tabla[columnas_vista].to_string(index=False),
        "",
        "Advertencia: las métricas de modelos con variables posteriores",
        "al evento no son directamente comparables con las preventivas.",
        "Confirmar la disponibilidad previa de cada variable antes de",
        "considerar un modelo apto para uso preventivo.",
    ]

    if errores:
        texto.extend(["", "ARCHIVOS NO PROCESADOS:"])
        texto.extend(errores)

    SALIDA_TXT.write_text("\n".join(texto), encoding="utf-8")

    print("=" * 75)
    print("CONSOLIDACIÓN CORREGIDA")
    print("=" * 75)
    print(tabla[columnas_vista].to_string(index=False))

    if errores:
        print("\nAvisos:")
        for error in errores:
            print("-", error)

    print("\nArchivos generados:")
    print(SALIDA_CSV)
    print(SALIDA_TXT)


if __name__ == "__main__":
    main()
