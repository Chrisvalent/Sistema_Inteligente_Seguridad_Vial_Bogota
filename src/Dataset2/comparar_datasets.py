import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_1_ORIGINAL = BASE_DIR / "data" / "raw" / "Historico_siniestros_bogota.xlsx"
DATASET_1_LIMPIO = BASE_DIR / "data" / "processed" / "siniestros_limpios.csv"
DATASET_2 = BASE_DIR / "data" / "processed" / "siniestros_integrados.csv"


print("=" * 70)
print("COMPARACIÓN DATASET 1 vs DATASET 2")
print("=" * 70)


# ============================================================
# CARGAR DATASET 1 ORIGINAL
# ============================================================

print("\nCargando Dataset 1 original...")

df1_original = pd.read_excel(
    DATASET_1_ORIGINAL,
    usecols=[
        "CODIGO_ACCIDENTE",
        "FECHA_OCURRENCIA_ACC",
        "GRAVEDAD",
        "LATITUD",
        "LONGITUD",
        "LOCALIDAD",
        "CIV",
        "PK_CALZADA"
    ]
)

print(f"Dataset 1 original: {len(df1_original):,} registros")


# ============================================================
# CARGAR DATASET 1 LIMPIO
# ============================================================

print("\nCargando Dataset 1 procesado...")

df1_limpio = pd.read_csv(DATASET_1_LIMPIO)

print(f"Dataset 1 procesado: {len(df1_limpio):,} registros")


# ============================================================
# CARGAR DATASET 2
# ============================================================

print("\nCargando Dataset 2 integrado...")

df2 = pd.read_csv(DATASET_2)

print(f"Dataset 2: {len(df2):,} registros")


# ============================================================
# CÓDIGOS DE ACCIDENTE
# ============================================================

print("\n" + "=" * 70)
print("CÓDIGOS DE ACCIDENTE")
print("=" * 70)

codigos1 = set(
    df1_original["CODIGO_ACCIDENTE"].dropna()
)

codigos2 = set(
    df2["CODIGO_ACCIDENTE"].dropna()
)

interseccion = codigos1.intersection(codigos2)

print(f"Dataset 1 únicos: {len(codigos1):,}")
print(f"Dataset 2 únicos: {len(codigos2):,}")
print(f"Accidentes presentes en ambos: {len(interseccion):,}")

porcentaje1 = len(interseccion) / len(codigos1) * 100
porcentaje2 = len(interseccion) / len(codigos2) * 100

print(f"Cobertura sobre Dataset 1: {porcentaje1:.2f}%")
print(f"Cobertura sobre Dataset 2: {porcentaje2:.2f}%")


# ============================================================
# CRUCE
# ============================================================

print("\n" + "=" * 70)
print("CRUCE DE INFORMACIÓN")
print("=" * 70)

df_cruce = df2.merge(
    df1_original,
    on="CODIGO_ACCIDENTE",
    how="inner",
    suffixes=("_D2", "_D1")
)

print(f"Registros después del cruce: {len(df_cruce):,}")


# ============================================================
# FECHAS
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN DE FECHAS")
print("=" * 70)

fecha_d2 = pd.to_datetime(
    df_cruce["FECHA"],
    errors="coerce"
)

fecha_d1 = pd.to_datetime(
    df_cruce["FECHA_OCURRENCIA_ACC"],
    errors="coerce"
)

validas = fecha_d2.notna() & fecha_d1.notna()

if validas.sum() > 0:

    iguales = (
        fecha_d2.dt.date == fecha_d1.dt.date
    )

    coincidencias = iguales[validas].sum()

    porcentaje = coincidencias / validas.sum() * 100

    print(f"Fechas comparables: {validas.sum():,}")
    print(f"Fechas iguales: {coincidencias:,}")
    print(f"Coincidencia: {porcentaje:.2f}%")


# ============================================================
# GRAVEDAD
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN DE GRAVEDAD")
print("=" * 70)

comparacion = pd.crosstab(
    df_cruce["GRAVEDAD_D2"],
    df_cruce["GRAVEDAD_D1"]
)

print(comparacion)


# ============================================================
# VARIABLES GEOGRÁFICAS
# ============================================================

print("\n" + "=" * 70)
print("COBERTURA GEOGRÁFICA DEL DATASET 1")
print("=" * 70)

columnas_geo = [
    "LATITUD",
    "LONGITUD",
    "LOCALIDAD",
    "CIV",
    "PK_CALZADA"
]

for columna in columnas_geo:

    disponibles = df_cruce[columna].notna().sum()

    porcentaje = (
        disponibles / len(df_cruce) * 100
    )

    print(
        f"{columna}: "
        f"{disponibles:,} disponibles "
        f"({porcentaje:.2f}%)"
    )


# ============================================================
# RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN FINALIZADA")
print("=" * 70)