import pandas as pd
import os


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

RUTA_ENTRADA = "data/raw/Historico_siniestros_bogota.xlsx"
RUTA_SALIDA = "data/processed/siniestros_limpios.csv"


# ==========================================================
# NORMALIZACIÓN DE LATITUD
# ==========================================================

def normalizar_latitud(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)

    # Ya está en formato normal
    if 4 <= abs(valor) <= 5:
        return valor

    # Mover el punto decimal hasta llegar al rango esperado
    while abs(valor) > 5:
        valor = valor / 10

    if 4 <= abs(valor) <= 5:
        return valor

    return None


# ==========================================================
# NORMALIZACIÓN DE LONGITUD
# ==========================================================

def normalizar_longitud(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)

    # Ya está en formato normal
    if 73 <= abs(valor) <= 75:
        return valor

    # Mover el punto decimal hasta llegar al rango esperado
    while abs(valor) > 75:
        valor = valor / 10

    if 73 <= abs(valor) <= 75:
        return valor

    return None


# ==========================================================
# CARGAR DATOS
# ==========================================================

print("=" * 60)
print("PROCESAMIENTO DEL DATASET DE SINIESTROS VIALES")
print("=" * 60)

print("\nCargando dataset original...")

df = pd.read_excel(RUTA_ENTRADA)

print(f"Registros originales: {len(df):,}")
print(f"Columnas originales: {len(df.columns)}")


# ==========================================================
# NORMALIZAR COORDENADAS
# ==========================================================

print("\nNormalizando coordenadas...")

df["LATITUD"] = df["LATITUD"].apply(normalizar_latitud)
df["LONGITUD"] = df["LONGITUD"].apply(normalizar_longitud)


# ==========================================================
# CONVERTIR FECHAS
# ==========================================================

print("Procesando fechas...")

df["FECHA_HORA_ACC"] = pd.to_datetime(
    df["FECHA_HORA_ACC"],
    errors="coerce",
    utc=True
)

df["FECHA_OCURRENCIA_ACC"] = pd.to_datetime(
    df["FECHA_OCURRENCIA_ACC"],
    errors="coerce",
    utc=True
)


# ==========================================================
# CREAR VARIABLES TEMPORALES
# ==========================================================

df["HORA"] = df["FECHA_HORA_ACC"].dt.hour

df["DIA_SEMANA"] = df["FECHA_HORA_ACC"].dt.dayofweek

df["MES"] = df["FECHA_HORA_ACC"].dt.month

df["DIA"] = df["FECHA_HORA_ACC"].dt.day

df["ES_FIN_DE_SEMANA"] = (
    df["DIA_SEMANA"] >= 5
).astype(int)


# ==========================================================
# ELIMINAR REGISTROS SIN VARIABLE OBJETIVO
# ==========================================================

df = df.dropna(subset=["GRAVEDAD"])


# ==========================================================
# TRATAMIENTO DE VARIABLES CATEGÓRICAS
# ==========================================================

df["LOCALIDAD"] = df["LOCALIDAD"].fillna("DESCONOCIDA")

df["CLASE_ACC"] = df["CLASE_ACC"].fillna("DESCONOCIDA")


# ==========================================================
# TRATAMIENTO DE VARIABLES NUMÉRICAS
# ==========================================================

# Indicadores de ausencia

df["CIV_FALTANTE"] = df["CIV"].isna().astype(int)

df["PK_CALZADA_FALTANTE"] = (
    df["PK_CALZADA"].isna()
).astype(int)


# Imputación

df["CIV"] = df["CIV"].fillna(-1)

df["PK_CALZADA"] = df["PK_CALZADA"].fillna(-1)


# ==========================================================
# ELIMINAR COLUMNAS QUE NO USAREMOS COMO PREDICTORES
# ==========================================================

columnas_eliminar = [
    "X",
    "Y",
    "OBJECTID",
    "FORMULARIO",
    "CODIGO_ACCIDENTE",
    "DIRECCION",
    "FECHA_OCURRENCIA_ACC",
    "FECHA_HORA_ACC"
]

df = df.drop(
    columns=columnas_eliminar,
    errors="ignore"
)


# ==========================================================
# ORDENAR COLUMNAS
# ==========================================================

columnas = [
    "GRAVEDAD",
    "CLASE_ACC",
    "LOCALIDAD",
    "ANO_OCURRENCIA_ACC",
    "MES",
    "DIA",
    "HORA",
    "DIA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "LATITUD",
    "LONGITUD",
    "CIV",
    "CIV_FALTANTE",
    "PK_CALZADA",
    "PK_CALZADA_FALTANTE"
]

df = df[columnas]


# ==========================================================
# ELIMINAR FILAS CON COORDENADAS NO VÁLIDAS
# ==========================================================

antes = len(df)

df = df.dropna(
    subset=["LATITUD", "LONGITUD"]
)

eliminados = antes - len(df)


# ==========================================================
# CREAR CARPETA DE SALIDA
# ==========================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================================
# GUARDAR DATASET PROCESADO
# ==========================================================

df.to_csv(
    RUTA_SALIDA,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# RESUMEN
# ==========================================================

print("\n" + "=" * 60)
print("PROCESAMIENTO COMPLETADO")
print("=" * 60)

print(f"\nRegistros finales: {len(df):,}")

print(f"Registros eliminados por coordenadas: {eliminados:,}")

print(f"Columnas finales: {len(df.columns)}")

print("\nColumnas:")

for columna in df.columns:
    print("-", columna)


print("\n===== DISTRIBUCIÓN DE GRAVEDAD =====")

print(df["GRAVEDAD"].value_counts())

print("\n===== PORCENTAJES =====")

print(
    df["GRAVEDAD"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n===== DATOS FALTANTES =====")

faltantes = df.isnull().sum()

print(
    faltantes[faltantes > 0]
)

print("\n===== ARCHIVO GENERADO =====")

print(RUTA_SALIDA)