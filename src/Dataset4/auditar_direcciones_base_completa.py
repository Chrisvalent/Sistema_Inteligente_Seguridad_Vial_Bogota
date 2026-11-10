
from pathlib import Path
import re
import pandas as pd

# ==========================================================
# CONFIGURACION
# ==========================================================

BASE = Path(__file__).resolve().parents[2]

ENTRADA = (
    BASE / "data" / "processed" / "siniestros_integrados.csv"
)

SALIDA_DETALLE = (
    BASE / "models" / "auditoria_direcciones_completa.csv"
)

SALIDA_RESUMEN = (
    BASE / "models" / "resumen_direcciones_completa.csv"
)

# Nomenclaturas habituales de Bogotá
TIPOS_VIA = r"KR|CR|CL|AC|AK|AV|TV|DG|DI"

# ==========================================================
# FUNCIONES
# ==========================================================

def normalizar(texto):
    if pd.isna(texto):
        return ""

    texto = str(texto).upper().strip()
    texto = re.sub(r"\s+", " ", texto)

    # Normalizar espacios alrededor de guiones
    texto = re.sub(r"\s*-\s*", "-", texto)

    return texto


def extraer_vias(direccion):
    """
    Extrae nomenclaturas numéricas reconocibles.
    No pretende geocodificar la dirección.
    """

    patron = (
        rf"(?<![A-Z0-9])"
        rf"({TIPOS_VIA})"
        rf"\s*"
        rf"(\d+[A-Z]?)"
        rf"\s*(S)?"
        rf"(?![A-Z0-9])"
    )

    coincidencias = re.findall(
        patron,
        direccion
    )

    vias = []

    for tipo, numero, sur in coincidencias:
        via = tipo + numero

        if sur:
            via += " S"

        vias.append(via)

    return list(dict.fromkeys(vias))


def clasificar_direccion(direccion):
    if not direccion:
        return "VACIA"

    vias = extraer_vias(direccion)

    if len(vias) >= 2:
        return "DOS_O_MAS_VIAS_NUMERICAS"

    if len(vias) == 1:
        return "UNA_VIA_NUMERICA"

    if any(
        palabra in direccion
        for palabra in [
            "AVENIDA",
            "AUTOPISTA",
            "CIRCUNVALAR",
            "AMERICAS",
            "BOYACA",
            "CARACAS",
            "SUBA",
            "CALI"
        ]
    ):
        return "VIA_CON_NOMBRE_ESPECIAL"

    return "NO_RECONOCIDA_AUTOMATICAMENTE"


# ==========================================================
# LECTURA
# ==========================================================

print("\n==============================================")
print(" AUDITORIA DE DIRECCIONES DE TODA LA BASE")
print("==============================================")

if not ENTRADA.exists():
    raise FileNotFoundError(
        f"No existe el archivo:\n{ENTRADA}"
    )

df = pd.read_csv(
    ENTRADA,
    encoding="utf-8-sig",
    low_memory=False
)

print(f"\nRegistros leidos: {len(df)}")

requeridas = [
    "DIRECCION",
    "CODIGO_LOCALIDAD"
]

faltantes = [
    c for c in requeridas
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas: {faltantes}"
    )

# ==========================================================
# NORMALIZACION
# ==========================================================

df["DIRECCION_NORMALIZADA"] = (
    df["DIRECCION"].apply(normalizar)
)

df["CODIGO_LOCALIDAD"] = (
    df["CODIGO_LOCALIDAD"].astype(str).str.strip()
)

df["VIAS_EXTRAIDAS"] = (
    df["DIRECCION_NORMALIZADA"]
    .apply(extraer_vias)
    .apply(lambda x: " | ".join(x))
)

df["CANTIDAD_VIAS_EXTRAIDAS"] = (
    df["DIRECCION_NORMALIZADA"]
    .apply(lambda x: len(extraer_vias(x)))
)

df["TIPO_DIRECCION"] = (
    df["DIRECCION_NORMALIZADA"]
    .apply(clasificar_direccion)
)

# ==========================================================
# RESUMEN POR DIRECCION Y LOCALIDAD
# ==========================================================

resumen = (
    df.groupby(
        [
            "CODIGO_LOCALIDAD",
            "DIRECCION_NORMALIZADA"
        ],
        dropna=False
    )
    .agg(
        TOTAL_SINIESTROS=("DIRECCION", "size"),
        DIRECCION_ORIGINAL=("DIRECCION", "first"),
        VIAS_EXTRAIDAS=("VIAS_EXTRAIDAS", "first"),
        CANTIDAD_VIAS_EXTRAIDAS=(
            "CANTIDAD_VIAS_EXTRAIDAS",
            "first"
        ),
        TIPO_DIRECCION=("TIPO_DIRECCION", "first")
    )
    .reset_index()
)

resumen = resumen.sort_values(
    "TOTAL_SINIESTROS",
    ascending=False
)

# ==========================================================
# EXPORTAR
# ==========================================================

df.to_csv(
    SALIDA_DETALLE,
    index=False,
    encoding="utf-8-sig"
)

resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# RESULTADOS
# ==========================================================

print("\n==============================================")
print(" RESULTADOS GENERALES")
print("==============================================")

print("Total de siniestros:", len(df))

print(
    "Direcciones originales diferentes:",
    df["DIRECCION"].nunique(dropna=True)
)

print(
    "Direcciones normalizadas diferentes:",
    df["DIRECCION_NORMALIZADA"].nunique()
)

print(
    "Combinaciones dirección-localidad:",
    len(resumen)
)

print("\nCLASIFICACION DE DIRECCIONES:")

print(
    df["TIPO_DIRECCION"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nTOP 20 DIRECCIONES MAS FRECUENTES:")

print(
    resumen[
        [
            "CODIGO_LOCALIDAD",
            "DIRECCION_ORIGINAL",
            "TOTAL_SINIESTROS",
            "VIAS_EXTRAIDAS",
            "TIPO_DIRECCION"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nDIRECCIONES SIN VIAS NUMERICAS RECONOCIDAS:")

sin_vias = resumen[
    resumen["CANTIDAD_VIAS_EXTRAIDAS"] == 0
]

print(
    sin_vias[
        [
            "CODIGO_LOCALIDAD",
            "DIRECCION_ORIGINAL",
            "TOTAL_SINIESTROS",
            "TIPO_DIRECCION"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nARCHIVOS GENERADOS:")
print(SALIDA_DETALLE)
print(SALIDA_RESUMEN)

print("\nNOTA:")
print(
    "La extraccion de nomenclatura no equivale a "
    "una georreferenciacion ni valida una interseccion."
)

print("\nProceso terminado.")
