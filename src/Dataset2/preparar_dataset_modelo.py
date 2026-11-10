import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_FILE = Path("data/processed/siniestros_enriquecido.csv")
OUTPUT_FILE = Path("data/processed/siniestros_modelo.csv")


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("PREPARACIÓN DEL DATASET PARA MACHINE LEARNING")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(
    f"\nDataset original: "
    f"{df.shape[0]:,} filas x {df.shape[1]} columnas"
)


# Guardamos la variable objetivo antes de eliminarla
target = df["GRAVEDAD"].copy()


# ============================================================
# 1. ELIMINAR VARIABLES CON FUGA DE INFORMACIÓN
# ============================================================

leakage_columns = [
    "GRAVEDAD",
    "CODIGO_ACCIDENTE",
    "NUM_ILESOS",
    "NUM_HERIDOS_ACTOR",
    "NUM_FALLECIDOS_ACTOR",
]

existing_leakage = [
    col for col in leakage_columns
    if col in df.columns
]

print("\nVariables eliminadas por fuga de información:")

for col in existing_leakage:
    print(f"  - {col}")

df = df.drop(columns=existing_leakage)


# ============================================================
# 2. PROCESAR FECHA
# ============================================================

print("\nProcesando FECHA...")

if "FECHA" in df.columns:

    fecha = pd.to_datetime(
        df["FECHA"],
        errors="coerce",
        dayfirst=True
    )

    fecha_variables = pd.DataFrame({
        "ANO": fecha.dt.year,
        "MES": fecha.dt.month,
        "DIA": fecha.dt.day,
        "DIA_SEMANA": fecha.dt.dayofweek,
        "ES_FIN_DE_SEMANA":
            (fecha.dt.dayofweek >= 5).astype(int)
    }, index=df.index)

    df = pd.concat(
        [
            df.drop(columns=["FECHA"]),
            fecha_variables
        ],
        axis=1
    )


# ============================================================
# 3. PROCESAR HORA
# ============================================================

print("Procesando HORA...")

if "HORA" in df.columns:

    hora_original = df["HORA"].astype(str)

    # Intentar convertir HH:MM:SS
    hora_td = pd.to_timedelta(
        hora_original,
        errors="coerce"
    )

    hora_num = (
        hora_td.dt.total_seconds() / 3600
    )

    # Para valores que no pudieron convertirse,
    # intentamos extraer la hora directamente.
    mask = hora_num.isna()

    if mask.any():

        hora_extraida = pd.to_numeric(
            hora_original
            .str.extract(r"^(\d{1,2})")[0],
            errors="coerce"
        )

        hora_num.loc[mask] = hora_extraida.loc[mask]

    df["HORA_NUM"] = hora_num


    # ========================================================
    # FRANJA HORARIA
    # ========================================================

    def clasificar_franja(hora):

        if pd.isna(hora):
            return "DESCONOCIDA"

        if 0 <= hora < 6:
            return "MADRUGADA"

        elif 6 <= hora < 12:
            return "MANANA"

        elif 12 <= hora < 18:
            return "TARDE"

        elif 18 <= hora < 24:
            return "NOCHE"

        return "DESCONOCIDA"


    df["FRANJA_HORARIA"] = hora_num.apply(
        clasificar_franja
    )

    df = df.drop(columns=["HORA"])


# ============================================================
# 4. ELIMINAR COLUMNAS DE TEXTO PROBLEMÁTICAS
# ============================================================

columns_to_drop = [
    "DIRECCION",
    "CHOQUE_DESCRIPCION",
    "OBJETO_FIJO_DESCRIPCION",
]

existing_drop = [
    col for col in columns_to_drop
    if col in df.columns
]

print(
    "\nColumnas excluidas por alta cardinalidad/redundancia:"
)

for col in existing_drop:
    print(f"  - {col}")

df = df.drop(columns=existing_drop)


# ============================================================
# 5. ELIMINAR VARIABLES CASI CONSTANTES
# ============================================================

print("\nBuscando variables casi constantes...")

near_constant = []

for col in df.columns:

    frecuencia = df[col].value_counts(
        dropna=False,
        normalize=True
    )

    if len(frecuencia) > 0:

        if frecuencia.iloc[0] >= 0.999:

            near_constant.append(col)


print(
    f"Variables casi constantes encontradas: "
    f"{len(near_constant)}"
)

for col in near_constant:
    print(f"  - {col}")

df = df.drop(columns=near_constant)


# ============================================================
# 6. VARIABLES CATEGÓRICAS
# ============================================================

categorical_columns = [
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
    "FRANJA_HORARIA",
]

categorical_columns = [
    col
    for col in categorical_columns
    if col in df.columns
]


# ============================================================
# 7. VARIABLES NUMÉRICAS
# ============================================================

numeric_columns = [
    col
    for col in df.columns
    if col not in categorical_columns
]


# ============================================================
# 8. PROCESAR CATEGÓRICAS
# ============================================================

print("\nProcesando variables categóricas...")

for col in categorical_columns:

    df[col] = df[col].astype(str)

    df[col] = df[col].replace(
        [
            "nan",
            "NaN",
            "None",
            "",
            "<NA>"
        ],
        "DESCONOCIDO"
    )

    df[col] = df[col].fillna(
        "DESCONOCIDO"
    )


# ============================================================
# 9. PROCESAR NUMÉRICAS
# ============================================================

print("Procesando variables numéricas...")

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if df[col].isna().any():

        mediana = df[col].median()

        if pd.isna(mediana):
            mediana = 0

        df[col] = df[col].fillna(
            mediana
        )


# ============================================================
# 10. LIMPIAR VARIABLE OBJETIVO
# ============================================================

target = (
    target
    .astype(str)
    .str.strip()
)

target = target.replace(
    [
        "nan",
        "None",
        ""
    ],
    np.nan
)

valid_target = target.notna()

df = df.loc[valid_target].copy()

target = target.loc[valid_target].copy()


# ============================================================
# 11. AGREGAR VARIABLE OBJETIVO
# ============================================================

df["GRAVEDAD"] = target.values


# ============================================================
# 12. REORDENAR
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col != "GRAVEDAD"
]

df = df[
    feature_columns + ["GRAVEDAD"]
]


# ============================================================
# 13. VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIÓN FINAL")
print("=" * 70)

print(
    f"\nFilas finales:    {df.shape[0]:,}"
)

print(
    f"Columnas finales: {df.shape[1]:,}"
)


# ============================================================
# DISTRIBUCIÓN DEL TARGET
# ============================================================

print("\nDistribución del objetivo:")

print(
    df["GRAVEDAD"].value_counts()
)

print("\nPorcentajes:")

print(
    (
        df["GRAVEDAD"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# DUPLICADOS
# ============================================================

print("\nDuplicados completos:")

print(
    df.duplicated().sum()
)


# ============================================================
# FALTANTES
# ============================================================

print("\nValores faltantes por columna:")

missing = df.isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No hay valores faltantes.")
else:
    print(missing)

print(
    f"\nTotal de valores faltantes: "
    f"{df.isna().sum().sum()}"
)


# ============================================================
# VARIABLES
# ============================================================

print("\nVariables categóricas:")

for col in categorical_columns:

    if col in df.columns:
        print(f"  - {col}")


print("\nVariables numéricas:")

for col in df.columns:

    if (
        col not in categorical_columns
        and col != "GRAVEDAD"
    ):
        print(f"  - {col}")


# ============================================================
# DISTRIBUCIÓN DE FRANJAS
# ============================================================

if "FRANJA_HORARIA" in df.columns:

    print("\nDistribución de franjas horarias:")

    print(
        df["FRANJA_HORARIA"]
        .value_counts()
    )


# ============================================================
# GUARDAR
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)

print("\nArchivo generado:")

print(OUTPUT_FILE)

print("\nDataset listo para Machine Learning.")