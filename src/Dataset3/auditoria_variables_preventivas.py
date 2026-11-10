
"""
Auditoría de variables preventivas del proyecto de seguridad vial.

No entrena modelos.
No modifica los modelos ni sus métricas existentes.
Genera un inventario de columnas para revisar su uso preventivo.
"""

from pathlib import Path
import json
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "siniestros_modelo.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "models"


# Variables que pueden conocerse antes del siniestro,
# siempre que representen información disponible al momento
# de realizar una predicción.
PREVENTIVAS = {
    "ANO",
    "ANIO",
    "AÑO",
    "MES",
    "DIA",
    "DÍA",
    "DIA_SEMANA",
    "DÍA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "HORA_NUM",
    "FRANJA_HORARIA",
    "CODIGO_LOCALIDAD",
    "COD_LOCALIDAD",
    "LOCALIDAD",
}

# Variables que describen el resultado o circunstancias
# observadas durante/después del siniestro.
POSTERIORES = {
    "GRAVEDAD",
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "NUM_ACTORES",
    "EDAD_PROMEDIO_ACTORES",
    "EDAD_MAX_ACTOR",
    "NUM_CONDUCTORES",
    "NUM_PEATONES",
    "NUM_VEHICULOS",
    "NUM_VEHICULOS_EN_FUGA",
    "NUM_CLASES_VEHICULO",
    "NUM_SERVICIOS",
    "NUM_MODALIDADES",
    "NUM_HIPOTESIS",
    "NUM_CAUSAS_DIFERENTES",
    "NUM_HIPOTESIS_FALTANTE",
    "CAUSA_CONDUCTOR",
    "CAUSA_VEHICULO",
    "CAUSA_VIA",
    "CAUSA_PEATON",
    "CAUSA_PASAJERO",
    "CAUSA_CICLISTA_MOTOCICLISTA",
}

# Variables que necesitan revisión de su definición y origen.
REVISAR = {
    "DISENO_LUGAR",
    "DISEÑO_LUGAR",
}


def normalizar(nombre):
    nombre = str(nombre).strip().upper()
    nombre = unicodedata.normalize("NFKD", nombre)
    return "".join(
        caracter
        for caracter in nombre
        if not unicodedata.combining(caracter)
    )


def clasificar_variable(nombre):
    original = str(nombre).strip().upper()
    limpio = normalizar(nombre)

    if limpio == "GRAVEDAD":
        return "OBJETIVO", (
            "Etiqueta que se busca predecir; no debe incluirse "
            "como variable de entrada."
        )

    if limpio in {normalizar(x) for x in PREVENTIVAS}:
        return "PREVENTIVA_CANDIDATA", (
            "Potencialmente disponible antes del evento; "
            "confirmar disponibilidad en el momento de predicción."
        )

    if limpio in {normalizar(x) for x in POSTERIORES}:
        return "POSTERIOR_AL_EVENTO", (
            "No usar como entrada en un modelo preventivo "
            "sin demostrar que la información existía previamente."
        )

    if limpio in {normalizar(x) for x in REVISAR}:
        return "REQUIERE_REVISION", (
            "Verificar definición, fuente y disponibilidad temporal "
            "antes de decidir si se puede utilizar."
        )

    if limpio.startswith("CAUSA_"):
        return "POSTERIOR_AL_EVENTO", (
            "Las causas o hipótesis registradas requieren "
            "investigación del evento; excluir del modelo preventivo."
        )

    if limpio.startswith("NUM_"):
        return "REQUIERE_REVISION", (
            "Verificar si el conteo existe antes del evento. "
            "Si se calcula a partir de actores o vehículos "
            "del siniestro, excluirlo."
        )

    return "REQUIERE_REVISION", (
        "No clasificada automáticamente; revisar su significado "
        "y cuándo estaría disponible."
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{DATA_PATH}\n"
            "Corrige DATA_PATH si tu CSV está en otra ubicación."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUDITORÍA DE VARIABLES PREVENTIVAS")
    print("=" * 70)
    print(f"Leyendo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"\nRegistros: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    inventario = []

    for columna in df.columns:
        serie = df[columna]
        categoria, justificacion = clasificar_variable(columna)

        valores = (
            serie.dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        inventario.append({
            "variable": columna,
            "clasificacion": categoria,
            "tipo_dato": str(serie.dtype),
            "nulos": int(serie.isna().sum()),
            "porcentaje_nulos": round(
                float(serie.isna().mean() * 100), 2
            ),
            "valores_unicos": int(serie.nunique(dropna=True)),
            "ejemplos": " | ".join(valores),
            "justificacion": justificacion,
        })

    resultado = pd.DataFrame(inventario)

    # Mostrar primero las variables que pueden servir
    # y después las que requieren revisión.
    orden = {
        "PREVENTIVA_CANDIDATA": 0,
        "OBJETIVO": 1,
        "REQUIERE_REVISION": 2,
        "POSTERIOR_AL_EVENTO": 3,
    }

    resultado["_orden"] = resultado["clasificacion"].map(orden)
    resultado = resultado.sort_values(
        ["_orden", "variable"]
    ).drop(columns="_orden")

    ruta_csv = OUTPUT_DIR / "auditoria_variables_preventivas.csv"
    ruta_json = OUTPUT_DIR / "auditoria_variables_preventivas.json"

    resultado.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    resumen = {
        "archivo_analizado": str(DATA_PATH),
        "registros": int(len(df)),
        "total_columnas": int(len(df.columns)),
        "resumen_clasificaciones": {
            str(clave): int(valor)
            for clave, valor in (
                resultado["clasificacion"].value_counts().items()
            )
        },
        "variables": inventario,
        "advertencia": (
            "La clasificación es preliminar y automática. "
            "Cada variable debe validarse con su diccionario de datos "
            "y su disponibilidad real antes del siniestro."
        ),
    }

    ruta_json.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRESUMEN DE CLASIFICACIONES")
    print("-" * 70)

    for categoria, cantidad in (
        resultado["clasificacion"].value_counts().items()
    ):
        print(f"{categoria}: {cantidad}")

    print("\nVARIABLES PREVENTIVAS CANDIDATAS")
    print("-" * 70)

    candidatas = resultado[
        resultado["clasificacion"] == "PREVENTIVA_CANDIDATA"
    ]

    for variable in candidatas["variable"]:
        print(f"- {variable}")

    print("\nVARIABLES QUE REQUIEREN REVISIÓN")
    print("-" * 70)

    por_revisar = resultado[
        resultado["clasificacion"] == "REQUIERE_REVISION"
    ]

    for variable in por_revisar["variable"]:
        print(f"- {variable}")

    print("\nArchivos generados:")
    print(ruta_csv)
    print(ruta_json)


if __name__ == "__main__":
    main()
