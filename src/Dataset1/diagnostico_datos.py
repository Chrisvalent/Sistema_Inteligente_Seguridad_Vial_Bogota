import pandas as pd

ruta = "data/raw/Historico_siniestros_bogota.xlsx"

df = pd.read_excel(ruta)

print("=" * 60)
print("DIAGNÓSTICO DE CALIDAD DEL DATASET")
print("=" * 60)

# --------------------------------------------------
# 1. VALORES ÚNICOS POR COLUMNA
# --------------------------------------------------

print("\n===== VALORES ÚNICOS =====")

for columna in df.columns:
    print(f"\n{columna}: {df[columna].nunique()} valores únicos")

# --------------------------------------------------
# 2. VALORES FALTANTES
# --------------------------------------------------

print("\n===== VALORES FALTANTES =====")

faltantes = df.isnull().sum()
faltantes = faltantes[faltantes > 0]

print(faltantes)

# --------------------------------------------------
# 3. PORCENTAJE DE FALTANTES
# --------------------------------------------------

print("\n===== PORCENTAJE DE FALTANTES =====")

porcentaje_faltantes = (
    df.isnull().mean() * 100
).round(2)

porcentaje_faltantes = porcentaje_faltantes[
    porcentaje_faltantes > 0
].sort_values(ascending=False)

print(porcentaje_faltantes)

# --------------------------------------------------
# 4. COORDENADAS
# --------------------------------------------------

print("\n===== ESTADÍSTICAS DE COORDENADAS =====")

print("\nLATITUD:")
print(df["LATITUD"].describe())

print("\nLONGITUD:")
print(df["LONGITUD"].describe())

# --------------------------------------------------
# 5. VALORES EXTREMOS DE COORDENADAS
# --------------------------------------------------

print("\n===== LATITUD: 10 VALORES MÁS PEQUEÑOS =====")
print(df["LATITUD"].nsmallest(10).to_string(index=False))

print("\n===== LATITUD: 10 VALORES MÁS GRANDES =====")
print(df["LATITUD"].nlargest(10).to_string(index=False))

print("\n===== LONGITUD: 10 VALORES MÁS PEQUEÑOS =====")
print(df["LONGITUD"].nsmallest(10).to_string(index=False))

print("\n===== LONGITUD: 10 VALORES MÁS GRANDES =====")
print(df["LONGITUD"].nlargest(10).to_string(index=False))

# --------------------------------------------------
# 6. CLASE DE ACCIDENTE
# --------------------------------------------------

print("\n===== CLASE_ACC =====")
print(df["CLASE_ACC"].value_counts())

# --------------------------------------------------
# 7. LOCALIDAD
# --------------------------------------------------

print("\n===== LOCALIDAD =====")
print(df["LOCALIDAD"].value_counts())

# --------------------------------------------------
# 8. GRAVEDAD
# --------------------------------------------------

print("\n===== GRAVEDAD =====")
print(df["GRAVEDAD"].value_counts())

# --------------------------------------------------
# 9. TIPOS DE DATOS
# --------------------------------------------------

print("\n===== TIPOS DE DATOS =====")
print(df.dtypes)