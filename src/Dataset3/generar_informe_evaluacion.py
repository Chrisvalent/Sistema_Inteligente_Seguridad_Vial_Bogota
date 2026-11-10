from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_MODELOS = RAIZ / "models"

ARCHIVO_CSV = CARPETA_MODELOS / "comparativa_modelos_consolidada.csv"
ARCHIVO_RESUMEN = CARPETA_MODELOS / "resumen_modelos_consolidado.txt"
ARCHIVO_JSON = CARPETA_MODELOS / "informe_evaluacion_tecnica.json"
ARCHIVO_MD = CARPETA_MODELOS / "informe_evaluacion_tecnica.md"

# ============================================================
# UTILIDADES
# ============================================================

def buscar_columna(df, posibles):
    """Busca una columna ignorando mayúsculas y espacios."""
    mapa = {str(c).strip().lower(): c for c in df.columns}

    for nombre in posibles:
        clave = nombre.strip().lower()
        if clave in mapa:
            return mapa[clave]

    return None


def valor_fila(fila, columna):
    if columna is None:
        return None

    valor = fila.get(columna)

    if pd.isna(valor):
        return None

    try:
        return float(valor)
    except (TypeError, ValueError):
        return str(valor)


def porcentaje(valor):
    if valor is None:
        return "No disponible"
    return f"{valor * 100:.2f} %"


def decimal(valor):
    if valor is None:
        return "No disponible"
    return f"{valor:.4f}"


# ============================================================
# LECTURA DE RESULTADOS
# ============================================================

def cargar_resultados():
    if not ARCHIVO_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de resultados:\n{ARCHIVO_CSV}\n"
            "Ejecuta primero consolidar_resultados_modelos.py."
        )

    df = pd.read_csv(ARCHIVO_CSV)

    if df.empty:
        raise ValueError("El CSV de resultados está vacío.")

    return df


def clasificar_modelo(nombre):
    nombre = str(nombre).lower()

    if "v9c" in nombre or "v9d" in nombre:
        return (
            "Candidato preventivo experimental",
            "Utiliza variables candidatas disponibles antes del accidente; "
            "debe verificarse la disponibilidad real y la definición de cada campo."
        )

    if "a_base" in nombre or "b_con_diseno" in nombre:
        return (
            "Comparación experimental",
            "Comparación controlada entre variables temporales y localidad, "
            "con y sin DISENO_LUGAR. No demuestra por sí sola disponibilidad preventiva."
        )

    if "v9b" in nombre:
        return (
            "Experimental con variables posteriores",
            "No debe considerarse preventivo sin una auditoría de variables."
        )

    if any(version in nombre for version in ["v5", "v6", "v7", "v8"]):
        return (
            "Experimental; revisar fuga de información",
            "No asumir uso preventivo hasta verificar las variables utilizadas."
        )

    return (
        "Clasificación pendiente",
        "Revisar variables, partición de datos y propósito del experimento."
    )


# ============================================================
# INFORME
# ============================================================

def main():
    df = cargar_resultados()

    col_version = buscar_columna(
        df, ["version", "modelo", "nombre_modelo"]
    )
    col_bloque = buscar_columna(
        df, ["bloque_metricas", "bloque", "conjunto"]
    )
    col_accuracy = buscar_columna(
        df, ["accuracy", "exactitud"]
    )
    col_balanced = buscar_columna(
        df, ["balanced_accuracy", "balanced accuracy"]
    )
    col_macro = buscar_columna(
        df, ["f1_macro", "macro_f1", "f1 macro"]
    )
    col_weighted = buscar_columna(
        df, ["f1_weighted", "weighted_f1", "f1 ponderado"]
    )
    col_precision = buscar_columna(
        df, ["precision_muertos", "precision_con_muertos"]
    )
    col_recall = buscar_columna(
        df, ["recall_muertos", "recall_con_muertos"]
    )
    col_f1muertos = buscar_columna(
        df, ["f1_muertos", "f1_con_muertos"]
    )

    if col_version is None:
        raise ValueError(
            "No se encontró la columna de versión/modelo. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    registros = []

    for _, fila in df.iterrows():
        nombre = str(fila[col_version])
        bloque = (
            str(fila[col_bloque])
            if col_bloque is not None
            and pd.notna(fila[col_bloque])
            else "principal"
        )

        tipo, observacion = clasificar_modelo(nombre)

        registros.append({
            "modelo": nombre,
            "bloque": bloque,
            "tipo": tipo,
            "accuracy": valor_fila(fila, col_accuracy),
            "balanced_accuracy": valor_fila(fila, col_balanced),
            "f1_macro": valor_fila(fila, col_macro),
            "f1_weighted": valor_fila(fila, col_weighted),
            "precision_muertos": valor_fila(fila, col_precision),
            "recall_muertos": valor_fila(fila, col_recall),
            "f1_muertos": valor_fila(fila, col_f1muertos),
            "observacion": observacion
        })

    # No se mezclan validación y prueba al buscar el mejor resultado.
    filas_prueba = [
        r for r in registros
        if "prueba" in r["bloque"].lower()
    ]

    candidatos_prueba = [
        r for r in filas_prueba
        if r["f1_muertos"] is not None
    ]

    mejor_f1_prueba = (
        max(candidatos_prueba, key=lambda r: r["f1_muertos"])
        if candidatos_prueba else None
    )

    candidatos_preventivos = [
        r for r in registros
        if "v9c" in r["modelo"].lower()
        or "v9d" in r["modelo"].lower()
    ]

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")

    informe = {
        "fecha_generacion": fecha,
        "proyecto": (
            "Sistema Inteligente para la Predicción del Nivel de Riesgo "
            "de Accidentes de Tránsito en Bogotá"
        ),
        "objetivo_modelado": (
            "Clasificar la severidad de siniestros registrados en Solo Daños, "
            "Con Heridos y Con Muertos. El modelo actual no estima por sí solo "
            "la probabilidad de que ocurra un accidente."
        ),
        "resultados": registros,
        "mejor_f1_muertos_en_bloques_prueba": mejor_f1_prueba,
        "candidatos_preventivos": candidatos_preventivos,
        "conclusiones": [
            (
                "La accuracy no basta para evaluar el sistema debido al "
                "desbalance entre las clases."
            ),
            (
                "Las métricas de modelos con variables posteriores al accidente "
                "no son evidencia de capacidad preventiva."
            ),
            (
                "V9C y V9D son candidatos experimentales con variables "
                "potencialmente disponibles antes del accidente; su "
                "disponibilidad real debe validarse."
            ),
            (
                "La clase Con Muertos requiere especial atención mediante "
                "precisión, recall, F1 y análisis de falsos positivos."
            ),
            (
                "La predicción debe apoyar la decisión y no automatizarla."
            )
        ],
        "limitaciones": [
            "La clasificación depende de siniestros registrados en el conjunto de datos.",
            "No se incluyen observaciones de lugares y momentos sin accidente para estimar ocurrencia.",
            "Los resultados no demuestran causalidad ni eficacia de una intervención preventiva.",
            "La disponibilidad futura de cada variable debe comprobarse en la fuente de datos.",
            "El desempeño observado no garantiza el desempeño en producción."
        ]
    }

    with open(ARCHIVO_JSON, "w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)

    lineas = [
        "# INFORME TÉCNICO DE EVALUACIÓN DE MODELOS",
        "",
        f"**Fecha de generación:** {fecha}",
        "",
        "**Proyecto:** Sistema Inteligente para la Predicción del Nivel "
        "de Riesgo de Accidentes de Tránsito en Bogotá",
        "",
        "## 1. Objetivo y alcance",
        "",
        "El objetivo actual es clasificar la severidad de siniestros "
        "registrados en las clases Solo Daños, Con Heridos y Con Muertos. "
        "No se estima la probabilidad de ocurrencia de accidentes, porque "
        "el conjunto actual está compuesto por registros de siniestros.",
        "",
        "## 2. Resultados consolidados",
        "",
        "| Modelo | Bloque | Tipo | Accuracy | Balanced accuracy | F1 macro | "
        "Precisión muertos | Recall muertos | F1 muertos |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"
    ]

    for r in registros:
        lineas.append(
            f"| {r['modelo']} | {r['bloque']} | {r['tipo']} | "
            f"{porcentaje(r['accuracy'])} | "
            f"{decimal(r['balanced_accuracy'])} | "
            f"{decimal(r['f1_macro'])} | "
            f"{porcentaje(r['precision_muertos'])} | "
            f"{porcentaje(r['recall_muertos'])} | "
            f"{decimal(r['f1_muertos'])} |"
        )

    lineas.extend([
        "",
        "## 3. Interpretación",
        "",
        "- La accuracy debe interpretarse junto con balanced accuracy y F1 macro.",
        "- La precisión de Con Muertos indica qué proporción de las predicciones "
        "de esa clase corresponde realmente a siniestros con muertos.",
        "- El recall de Con Muertos indica qué proporción de los siniestros "
        "fatales reales fue identificada.",
        "- Un recall bajo implica que muchos siniestros fatales no se identifican.",
        "- Una precisión baja implica un número elevado de falsas alarmas.",
        "",
        "## 4. Consideraciones sobre prevención",
        "",
        "Los modelos V9C y V9D utilizan variables candidatas previas al evento. "
        "Antes de emplearlos preventivamente se debe confirmar que cada variable "
        "está disponible antes del siniestro y que su significado es consistente.",
        "",
        "Los modelos experimentales que incorporan información posterior al "
        "accidente no deben presentarse como modelos preventivos.",
        "",
        "## 5. Limitaciones",
        "",
        "- El conjunto actual contiene siniestros registrados.",
        "- No permite estimar directamente la probabilidad de ocurrencia de un accidente.",
        "- Los resultados no prueban causalidad ni eficacia preventiva.",
        "- Se necesita validar disponibilidad, calidad y estabilidad temporal de las variables.",
        "- Las métricas obtenidas no garantizan el mismo desempeño en producción.",
        "",
        "## 6. Conclusión",
        "",
        "Los experimentos permiten comparar estrategias de clasificación, pero "
        "el desempeño de los candidatos preventivos es todavía limitado, "
        "especialmente para la clase Con Muertos. No se recomienda afirmar que "
        "el modelo está listo para una aplicación preventiva sin validaciones "
        "adicionales.",
        "",
        "**Principio del proyecto:** La predicción apoya la decisión. No la automatiza.",
        ""
    ])

    with open(ARCHIVO_MD, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas))

    print("=" * 75)
    print("INFORME TÉCNICO GENERADO")
    print("=" * 75)
    print(f"CSV de entrada: {ARCHIVO_CSV}")
    print(f"Informe Markdown: {ARCHIVO_MD}")
    print(f"Informe JSON: {ARCHIVO_JSON}")

    if mejor_f1_prueba:
        print("\nMejor F1 de Con Muertos entre bloques de prueba disponibles:")
        print(f"Modelo: {mejor_f1_prueba['modelo']}")
        print(f"Bloque: {mejor_f1_prueba['bloque']}")
        print(f"F1: {decimal(mejor_f1_prueba['f1_muertos'])}")
        print(
            "Nota: este resultado es comparativo; no implica que el modelo "
            "sea preventivo."
        )
    else:
        print(
            "\nNo se pudo identificar un bloque de prueba con F1 de Con Muertos."
        )

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()
