import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARCHIVO = BASE_DIR / "data" / "processed" / "siniestros_integrados.csv"


# ============================================================
# CARGAR DATASET
# ============================================================

print("=" * 70)
print("VALIDACIÓN DEL DATASET INTEGRADO")
print("=" * 70)

print("\nCargando dataset...")

df = pd.read_csv(ARCHIVO)

print(f"Archivo: {ARCHIVO}")
print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 1. INFORMACIÓN GENERAL
# ============================================================

print("\n" + "=" * 70)
print("1. INFORMACIÓN GENERAL")
print("=" * 70)

print("\nTipos de datos:")
print(df.dtypes)


# ============================================================
# 2. DUPLICADOS
# ============================================================

print("\n" + "=" * 70)
print("2. DUPLICADOS")
print("=" * 70)

duplicados_codigo = df["CODIGO_ACCIDENTE"].duplicated().sum()
duplicados_filas = df.duplicated().sum()

print(f"Duplicados CODIGO_ACCIDENTE: {duplicados_codigo:,}")
print(f"Filas completamente duplicadas: {duplicados_filas:,}")


# ============================================================
# 3. VARIABLE OBJETIVO
# ============================================================

print("\n" + "=" * 70)
print("3. VARIABLE OBJETIVO: GRAVEDAD")
print("=" * 70)

print("\nFrecuencia:")
print(df["GRAVEDAD"].value_counts(dropna=False))

print("\nPorcentaje:")
print(
    (df["GRAVEDAD"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)


# ============================================================
# 4. CLASE DEL ACCIDENTE
# ============================================================

print("\n" + "=" * 70)
print("4. CLASE DEL ACCIDENTE")
print("=" * 70)

print(df["CLASE"].value_counts(dropna=False))


# ============================================================
# 5. CHOQUE
# ============================================================

print("\n" + "=" * 70)
print("5. TIPO DE CHOQUE")
print("=" * 70)

print(df["CHOQUE"].value_counts(dropna=False))


# ============================================================
# 6. OBJETO FIJO
# ============================================================

print("\n" + "=" * 70)
print("6. OBJETO FIJO")
print("=" * 70)

print(df["OBJETO_FIJO"].value_counts(dropna=False))


# ============================================================
# 7. DISEÑO DEL LUGAR
# ============================================================

print("\n" + "=" * 70)
print("7. DISEÑO DEL LUGAR")
print("=" * 70)

print(df["DISENO_LUGAR"].value_counts(dropna=False).head(30))


# ============================================================
# 8. LOCALIDADES
# ============================================================

print("\n" + "=" * 70)
print("8. CÓDIGO DE LOCALIDAD")
print("=" * 70)

print(f"Localidades diferentes: {df['CODIGO_LOCALIDAD'].nunique(dropna=True)}")

print("\nFrecuencia:")
print(df["CODIGO_LOCALIDAD"].value_counts(dropna=False).head(30))


# ============================================================
# 9. VARIABLES DE ACTORES
# ============================================================

print("\n" + "=" * 70)
print("9. VARIABLES DE ACTORES")
print("=" * 70)

columnas_actores = [
    "NUM_ACTORES",
    "EDAD_PROMEDIO_ACTORES",
    "EDAD_MAX_ACTOR",
    "NUM_CONDUCTORES",
    "NUM_PEATONES",
    "NUM_ILESOS",
    "NUM_HERIDOS_ACTOR",
    "NUM_FALLECIDOS_ACTOR"
]

print(df[columnas_actores].describe())


# ============================================================
# 10. VARIABLES DE VEHÍCULOS
# ============================================================

print("\n" + "=" * 70)
print("10. VARIABLES DE VEHÍCULOS")
print("=" * 70)

columnas_vehiculos = [
    "NUM_VEHICULOS",
    "NUM_VEHICULOS_EN_FUGA",
    "NUM_CLASES_VEHICULO",
    "NUM_SERVICIOS",
    "NUM_MODALIDADES"
]

print(df[columnas_vehiculos].describe())


# ============================================================
# 11. VARIABLES DE HIPÓTESIS
# ============================================================

print("\n" + "=" * 70)
print("11. VARIABLES DE HIPÓTESIS")
print("=" * 70)

columnas_hipotesis = [
    "NUM_HIPOTESIS",
    "NUM_CAUSAS_DIFERENTES"
]

print(df[columnas_hipotesis].describe())


# ============================================================
# 12. DATOS FALTANTES
# ============================================================

print("\n" + "=" * 70)
print("12. DATOS FALTANTES")
print("=" * 70)

faltantes = df.isnull().sum()

faltantes = faltantes[faltantes > 0].sort_values(ascending=False)

if len(faltantes) == 0:
    print("No existen valores faltantes.")
else:
    print(faltantes)


# ============================================================
# 13. VALORES ÚNICOS DE VARIABLES CATEGÓRICAS
# ============================================================

print("\n" + "=" * 70)
print("13. VARIABLES CATEGÓRICAS")
print("=" * 70)

categoricas = [
    "GRAVEDAD",
    "CLASE",
    "CHOQUE_DESCRIPCION",
    "OBJETO_FIJO_DESCRIPCION",
    "DISENO_LUGAR"
]

for columna in categoricas:

    print(f"\n--- {columna} ---")

    valores = df[columna].value_counts(dropna=False)

    print(valores.head(30))


# ============================================================
# 14. DETECCIÓN DE POSIBLE TARGET LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("14. POSIBLE TARGET LEAKAGE")
print("=" * 70)

print("""
Variables relacionadas directamente con el resultado del accidente:

- NUM_ILESOS
- NUM_HERIDOS_ACTOR
- NUM_FALLECIDOS_ACTOR

Estas variables NO deben utilizarse inicialmente como
predictores del modelo porque pueden revelar directamente
la gravedad real del accidente.
""")


# ============================================================
# 15. PROPUESTA INICIAL DE VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("15. PROPUESTA INICIAL DE VARIABLES")
print("=" * 70)

variables_candidato = [
    "FECHA",
    "HORA",
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
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
    "NUM_CAUSAS_DIFERENTES"
]

print("\nVariables candidatas:")
for columna in variables_candidato:
    print(f"- {columna}")


# ============================================================
# 16. VARIABLES A EXCLUIR
# ============================================================

print("\n" + "=" * 70)
print("16. VARIABLES A EXCLUIR INICIALMENTE")
print("=" * 70)

excluir = [
    "CODIGO_ACCIDENTE",
    "GRAVEDAD",
    "NUM_ILESOS",
    "NUM_HERIDOS_ACTOR",
    "NUM_FALLECIDOS_ACTOR"
]

for columna in excluir:
    print(f"- {columna}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIÓN FINALIZADA")
print("=" * 70)