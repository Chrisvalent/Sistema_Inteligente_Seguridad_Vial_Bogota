import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RUTA_DATASET = BASE_DIR / "data" / "processed" / "siniestros_enriquecido.csv"


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("VALIDACIÓN DEL DATASET ENRIQUECIDO")
print("=" * 70)

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATASET, low_memory=False)

print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 1. ESTRUCTURA GENERAL
# ============================================================

print("\n" + "=" * 70)
print("1. ESTRUCTURA GENERAL")
print("=" * 70)

print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]:,}")

print(f"\nDuplicados completos: {df.duplicated().sum():,}")

if "CODIGO_ACCIDENTE" in df.columns:
    print(
        f"Duplicados CODIGO_ACCIDENTE: "
        f"{df['CODIGO_ACCIDENTE'].duplicated().sum():,}"
    )


# ============================================================
# 2. VARIABLE OBJETIVO
# ============================================================

print("\n" + "=" * 70)
print("2. VARIABLE OBJETIVO: GRAVEDAD")
print("=" * 70)

if "GRAVEDAD" in df.columns:

    print(df["GRAVEDAD"].value_counts(dropna=False))

    print("\nPorcentaje:")
    print(
        (df["GRAVEDAD"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )

    print(f"\nValores faltantes: {df['GRAVEDAD'].isna().sum():,}")

else:
    print("ERROR: No existe la columna GRAVEDAD.")


# ============================================================
# 3. VALORES FALTANTES
# ============================================================

print("\n" + "=" * 70)
print("3. VALORES FALTANTES")
print("=" * 70)

faltantes = df.isna().sum()

faltantes = faltantes[faltantes > 0].sort_values(ascending=False)

if len(faltantes) == 0:
    print("No existen valores faltantes.")
else:

    tabla_faltantes = pd.DataFrame({
        "Faltantes": faltantes,
        "Porcentaje": (faltantes / len(df) * 100).round(2)
    })

    print(tabla_faltantes.to_string())


# ============================================================
# 4. TIPOS DE DATOS
# ============================================================

print("\n" + "=" * 70)
print("4. TIPOS DE DATOS")
print("=" * 70)

print(df.dtypes.to_string())


# ============================================================
# 5. VARIABLES CATEGÓRICAS
# ============================================================

print("\n" + "=" * 70)
print("5. VARIABLES CATEGÓRICAS")
print("=" * 70)

categoricas = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print(f"Cantidad: {len(categoricas)}")

for col in categoricas:
    print(f" - {col}: {df[col].nunique(dropna=False)} valores")


# ============================================================
# 6. VARIABLES NUMÉRICAS
# ============================================================

print("\n" + "=" * 70)
print("6. VARIABLES NUMÉRICAS")
print("=" * 70)

numericas = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print(f"Cantidad: {len(numericas)}")

for col in numericas:
    print(f" - {col}")


# ============================================================
# 7. VARIABLES CON POCA VARIACIÓN
# ============================================================

print("\n" + "=" * 70)
print("7. VARIABLES CON POCA VARIACIÓN")
print("=" * 70)

print(
    "Se identifican variables donde casi todos los registros "
    "tienen el mismo valor."
)

variables_poca_variacion = []

for col in df.columns:

    if col == "GRAVEDAD":
        continue

    frecuencia = df[col].value_counts(normalize=True, dropna=False)

    if len(frecuencia) > 0 and frecuencia.iloc[0] >= 0.99:
        variables_poca_variacion.append(
            (
                col,
                df[col].nunique(dropna=False),
                round(frecuencia.iloc[0] * 100, 2),
                frecuencia.index[0]
            )
        )

if variables_poca_variacion:

    tabla = pd.DataFrame(
        variables_poca_variacion,
        columns=[
            "Variable",
            "Valores_unicos",
            "Mayor_frecuencia_% ",
            "Valor_principal"
        ]
    )

    print(tabla.to_string(index=False))

else:
    print("No se encontraron variables con >=99% de un mismo valor.")


# ============================================================
# 8. VARIABLES DE POSIBLE DATA LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("8. VARIABLES DE POSIBLE DATA LEAKAGE")
print("=" * 70)

posible_leakage = [
    "NUM_ILESOS",
    "NUM_HERIDOS_ACTOR",
    "NUM_FALLECIDOS_ACTOR",
    "GRAVEDAD"
]

for col in posible_leakage:

    if col in df.columns:

        print(
            f" - {col}: PRESENTE "
            f"-> revisar antes de utilizarla como predictor"
        )


# ============================================================
# 9. VARIABLES DE IDENTIFICACIÓN
# ============================================================

print("\n" + "=" * 70)
print("9. VARIABLES DE IDENTIFICACIÓN")
print("=" * 70)

identificadores = [
    "CODIGO_ACCIDENTE",
    "CODIGO_ACCIDENTADO",
    "OBJECTID",
    "FORMULARIO"
]

for col in identificadores:

    if col in df.columns:
        print(f" - {col}: presente")


# ============================================================
# 10. VARIABLES DE ENRIQUECIMIENTO
# ============================================================

print("\n" + "=" * 70)
print("10. VARIABLES DE ENRIQUECIMIENTO")
print("=" * 70)

patrones = [
    "NUM_",
    "CAUSA_"
]

enriquecidas = []

for col in df.columns:

    if any(col.startswith(p) for p in patrones):
        enriquecidas.append(col)

print(f"Variables de enriquecimiento: {len(enriquecidas)}")

for col in enriquecidas:
    print(f" - {col}")


# ============================================================
# 11. DISTRIBUCIÓN DE VARIABLES DE ENRIQUECIMIENTO
# ============================================================

print("\n" + "=" * 70)
print("11. DISTRIBUCIÓN DE VARIABLES DE ENRIQUECIMIENTO")
print("=" * 70)

for col in enriquecidas:

    if pd.api.types.is_numeric_dtype(df[col]):

        print(
            f"{col}: "
            f"min={df[col].min()}, "
            f"max={df[col].max()}, "
            f"media={df[col].mean():.3f}"
        )


# ============================================================
# 12. COLUMNAS CANDIDATAS PARA EL MODELO
# ============================================================

print("\n" + "=" * 70)
print("12. COLUMNAS QUE NO DEBEN ENTRAR DIRECTAMENTE AL MODELO")
print("=" * 70)

excluir = set(
    [
        "GRAVEDAD",
        "CODIGO_ACCIDENTE",
        "CODIGO_ACCIDENTADO",
        "OBJECTID",
        "FORMULARIO",

        # Posible fuga de información
        "NUM_ILESOS",
        "NUM_HERIDOS_ACTOR",
        "NUM_FALLECIDOS_ACTOR"
    ]
)

presentes_excluir = [
    col for col in df.columns
    if col in excluir
]

for col in presentes_excluir:
    print(f" - {col}")


# ============================================================
# 13. CANDIDATAS INICIALES
# ============================================================

candidatas = [
    col for col in df.columns
    if col not in excluir
]

print("\n" + "=" * 70)
print("13. VARIABLES CANDIDATAS INICIALES")
print("=" * 70)

print(f"Cantidad de candidatas: {len(candidatas)}")

for col in candidatas:
    print(f" - {col}")


# ============================================================
# 14. RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN DE VALIDACIÓN")
print("=" * 70)

print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")
print(f"Variables categóricas: {len(categoricas)}")
print(f"Variables numéricas: {len(numericas)}")
print(f"Variables de enriquecimiento: {len(enriquecidas)}")
print(f"Variables candidatas iniciales: {len(candidatas)}")

print("\nDataset enriquecido cargado correctamente.")

print("\nIMPORTANTE:")
print(
    "Las variables candidatas son una primera selección. "
    "Antes de entrenar el modelo se realizará una revisión "
    "de disponibilidad temporal, fuga de información y "
    "calidad estadística."
)

print("\n" + "=" * 70)
print("VALIDACIÓN FINALIZADA")
print("=" * 70)