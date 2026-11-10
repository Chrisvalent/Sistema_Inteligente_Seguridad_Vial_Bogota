import pandas as pd

ruta = "data/raw/Historico_siniestros_bogota.xlsx"

df = pd.read_excel(ruta)

print("===== DISTRIBUCIÓN DE GRAVEDAD =====")

print(df["GRAVEDAD"].value_counts())

print("\n===== PORCENTAJES =====")

print(
    df["GRAVEDAD"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n===== VALORES ÚNICOS DE GRAVEDAD =====")

print(df["GRAVEDAD"].unique())

print("\n===== CLASE DE ACCIDENTE =====")

print(df["CLASE_ACC"].value_counts())

print("\n===== LOCALIDADES =====")

print(df["LOCALIDAD"].value_counts())

print("\n===== AÑOS =====")

print(df["ANO_OCURRENCIA_ACC"].value_counts().sort_index())

print("\n===== FECHA MÍNIMA Y MÁXIMA =====")

print("Desde:", df["FECHA_OCURRENCIA_ACC"].min())
print("Hasta:", df["FECHA_OCURRENCIA_ACC"].max())