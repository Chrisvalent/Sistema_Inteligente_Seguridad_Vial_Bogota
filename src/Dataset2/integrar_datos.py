import pandas as pd
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_ARCHIVO = (
    "data/raw/"
    "siniestros_viales_consolidados_bogota_dc.xlsx"
)

RUTA_SALIDA = (
    "data/processed/"
    "siniestros_integrados.csv"
)


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("INTEGRACIÓN - SINIESTROS VIALES BOGOTÁ")
print("=" * 70)


# ============================================================
# 1. CARGAR HOJAS
# ============================================================

print("\nCargando hojas...")

siniestros = pd.read_excel(
    RUTA_ARCHIVO,
    sheet_name="SINIESTROS"
)

actores = pd.read_excel(
    RUTA_ARCHIVO,
    sheet_name="ACTOR_VIAL"
)

vehiculos = pd.read_excel(
    RUTA_ARCHIVO,
    sheet_name="VEHICULOS"
)

hipotesis = pd.read_excel(
    RUTA_ARCHIVO,
    sheet_name="HIPOTESIS"
)

diccionario = pd.read_excel(
    RUTA_ARCHIVO,
    sheet_name="DICCIONARIO"
)


print("\nRegistros cargados:")

print(f"SINIESTROS:  {len(siniestros):,}")
print(f"ACTOR_VIAL:  {len(actores):,}")
print(f"VEHICULOS:   {len(vehiculos):,}")
print(f"HIPOTESIS:   {len(hipotesis):,}")


# ============================================================
# 2. LIMPIAR NOMBRES DE COLUMNAS
# ============================================================

for df in [
    siniestros,
    actores,
    vehiculos,
    hipotesis,
    diccionario
]:
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )


# ============================================================
# 3. FUNCIÓN PARA OBTENER DESCRIPCIONES
# ============================================================

def obtener_diccionario(
    hoja,
    campo
):

    tabla = diccionario[
        (diccionario["HOJA"] == hoja) &
        (diccionario["CAMPO"] == campo)
    ].copy()

    tabla = tabla[
        ["CODIGO", "DESCRIPCION"]
    ]

    tabla = tabla.drop_duplicates(
        subset=["CODIGO"]
    )

    return dict(
        zip(
            tabla["CODIGO"],
            tabla["DESCRIPCION"]
        )
    )


# ============================================================
# 4. TRADUCIR GRAVEDAD
# ============================================================

print("\n" + "=" * 70)
print("TRADUCIENDO VARIABLES CON EL DICCIONARIO OFICIAL")
print("=" * 70)

dic_gravedad = obtener_diccionario(
    "SINIESTROS",
    "GRAVEDAD"
)

dic_clase = obtener_diccionario(
    "SINIESTROS",
    "CLASE"
)

dic_choque = obtener_diccionario(
    "SINIESTROS",
    "CHOQUE"
)

dic_objeto = obtener_diccionario(
    "SINIESTROS",
    "OBJETO_FIJO"
)

print("\nCódigos de GRAVEDAD:")

for codigo, descripcion in dic_gravedad.items():
    print(
        f"{codigo} -> {descripcion}"
    )


# ============================================================
# 5. CONVERTIR GRAVEDAD Y CLASE
# ============================================================

siniestros["GRAVEDAD"] = (
    siniestros["GRAVEDAD"]
    .map(dic_gravedad)
)

siniestros["CLASE"] = (
    siniestros["CLASE"]
    .map(dic_clase)
)

siniestros["CHOQUE_DESCRIPCION"] = (
    siniestros["CHOQUE"]
    .map(dic_choque)
)

siniestros["OBJETO_FIJO_DESCRIPCION"] = (
    siniestros["OBJETO_FIJO"]
    .map(dic_objeto)
)


# ============================================================
# 6. AGREGAR ACTORES VIALES
# ============================================================

print("\n" + "=" * 70)
print("AGREGANDO ACTORES VIALES")
print("=" * 70)

actores["EDAD"] = pd.to_numeric(
    actores["EDAD"],
    errors="coerce"
)

actores["ES_CONDUCTOR"] = (
    actores["CONDICION"]
    .astype(str)
    .str.upper()
    .eq("CONDUCTOR")
    .astype(int)
)

actores["ES_PEATON"] = (
    actores["CONDICION"]
    .astype(str)
    .str.upper()
    .eq("PEATON")
    .astype(int)
)

actores["ES_ILESO"] = (
    actores["ESTADO"]
    .astype(str)
    .str.upper()
    .eq("ILESO")
    .astype(int)
)

actores["ES_HERIDO"] = (
    actores["ESTADO"]
    .astype(str)
    .str.upper()
    .str.contains("HERID")
    .astype(int)
)

actores["ES_FALLECIDO"] = (
    actores["ESTADO"]
    .astype(str)
    .str.upper()
    .str.contains(
        "MUER|FALLEC"
    )
    .astype(int)
)


actores_agregados = (
    actores
    .groupby("CODIGO_ACCIDENTE")
    .agg(
        NUM_ACTORES=(
            "CODIGO_ACCIDENTADO",
            "count"
        ),
        EDAD_PROMEDIO_ACTORES=(
            "EDAD",
            "mean"
        ),
        EDAD_MAX_ACTOR=(
            "EDAD",
            "max"
        ),
        NUM_CONDUCTORES=(
            "ES_CONDUCTOR",
            "sum"
        ),
        NUM_PEATONES=(
            "ES_PEATON",
            "sum"
        ),
        NUM_ILESOS=(
            "ES_ILESO",
            "sum"
        ),
        NUM_HERIDOS_ACTOR=(
            "ES_HERIDO",
            "sum"
        ),
        NUM_FALLECIDOS_ACTOR=(
            "ES_FALLECIDO",
            "sum"
        )
    )
    .reset_index()
)


print(
    f"\nAccidentes con información de actores: "
    f"{len(actores_agregados):,}"
)


# ============================================================
# 7. AGREGAR VEHÍCULOS
# ============================================================

print("\n" + "=" * 70)
print("AGREGANDO VEHÍCULOS")
print("=" * 70)

vehiculos["CLASE"] = pd.to_numeric(
    vehiculos["CLASE"],
    errors="coerce"
)

vehiculos["SERVICIO"] = pd.to_numeric(
    vehiculos["SERVICIO"],
    errors="coerce"
)

vehiculos["MODALIDAD"] = pd.to_numeric(
    vehiculos["MODALIDAD"],
    errors="coerce"
)

vehiculos["ENFUGA_BIN"] = (
    vehiculos["ENFUGA"]
    .astype(str)
    .str.upper()
    .eq("S")
    .astype(int)
)


vehiculos_agregados = (
    vehiculos
    .groupby("CODIGO_ACCIDENTE")
    .agg(
        NUM_VEHICULOS=(
            "VEHICULO",
            "count"
        ),
        NUM_VEHICULOS_EN_FUGA=(
            "ENFUGA_BIN",
            "sum"
        ),
        NUM_CLASES_VEHICULO=(
            "CLASE",
            "nunique"
        ),
        NUM_SERVICIOS=(
            "SERVICIO",
            "nunique"
        ),
        NUM_MODALIDADES=(
            "MODALIDAD",
            "nunique"
        )
    )
    .reset_index()
)


print(
    f"\nAccidentes con información de vehículos: "
    f"{len(vehiculos_agregados):,}"
)


# ============================================================
# 8. AGREGAR HIPÓTESIS
# ============================================================

print("\n" + "=" * 70)
print("AGREGANDO HIPÓTESIS")
print("=" * 70)


hipotesis_agregadas = (
    hipotesis
    .groupby("CODIGO_ACCIDENTE")
    .agg(
        NUM_HIPOTESIS=(
            "CODIGO_CAUSA",
            "count"
        ),
        NUM_CAUSAS_DIFERENTES=(
            "CODIGO_CAUSA",
            "nunique"
        )
    )
    .reset_index()
)


print(
    f"\nAccidentes con información de hipótesis: "
    f"{len(hipotesis_agregadas):,}"
)


# ============================================================
# 9. INTEGRACIÓN
# ============================================================

print("\n" + "=" * 70)
print("INTEGRANDO TABLAS")
print("=" * 70)


df = siniestros.merge(
    actores_agregados,
    on="CODIGO_ACCIDENTE",
    how="left"
)

print(
    f"Después de ACTOR_VIAL: "
    f"{len(df):,} registros"
)


df = df.merge(
    vehiculos_agregados,
    on="CODIGO_ACCIDENTE",
    how="left"
)

print(
    f"Después de VEHICULOS: "
    f"{len(df):,} registros"
)


df = df.merge(
    hipotesis_agregadas,
    on="CODIGO_ACCIDENTE",
    how="left"
)

print(
    f"Después de HIPOTESIS: "
    f"{len(df):,} registros"
)


# ============================================================
# 10. INDICADORES DE INFORMACIÓN FALTANTE
# ============================================================

print("\n" + "=" * 70)
print("INDICADORES DE INFORMACIÓN FALTANTE")
print("=" * 70)


columnas_agregadas = [
    "NUM_ACTORES",
    "NUM_VEHICULOS",
    "NUM_HIPOTESIS"
]


for columna in columnas_agregadas:

    columna_faltante = (
        columna + "_FALTANTE"
    )

    df[columna_faltante] = (
        df[columna]
        .isna()
        .astype(int)
    )


# ============================================================
# 11. GUARDAR
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO DATASET INTEGRADO")
print("=" * 70)


os.makedirs(
    "data/processed",
    exist_ok=True
)


df.to_csv(
    RUTA_SALIDA,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. VALIDACIONES
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIONES")
print("=" * 70)


print(
    f"\nRegistros finales: "
    f"{len(df):,}"
)

print(
    f"Columnas finales: "
    f"{len(df.columns)}"
)

print(
    f"Códigos de accidente únicos: "
    f"{df['CODIGO_ACCIDENTE'].nunique():,}"
)


duplicados = (
    df["CODIGO_ACCIDENTE"]
    .duplicated()
    .sum()
)

print(
    f"Duplicados de CODIGO_ACCIDENTE: "
    f"{duplicados:,}"
)


print("\nDistribución de GRAVEDAD:")

print(
    df["GRAVEDAD"]
    .value_counts(dropna=False)
)


print("\nDatos faltantes:")

faltantes = df.isnull().sum()

faltantes = faltantes[
    faltantes > 0
]

print(faltantes)


# ============================================================
# 13. COLUMNAS RESULTANTES
# ============================================================

print("\n" + "=" * 70)
print("COLUMNAS DEL DATASET INTEGRADO")
print("=" * 70)

for columna in df.columns:
    print("-", columna)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("INTEGRACIÓN FINALIZADA")
print("=" * 70)

print(
    f"\nArchivo generado:\n{RUTA_SALIDA}"
)