import pandas as pd

ruta = "data/raw/Historico_siniestros_bogota.xlsx"

df = pd.read_excel(ruta)

print("=" * 60)
print("ANÁLISIS DE COORDENADAS")
print("=" * 60)

# --------------------------------------------------
# LATITUD
# --------------------------------------------------

print("\n===== LATITUD =====")

print("\nValores originales:")
print(df["LATITUD"].describe())

print("\nCantidad con latitud normal (entre 4 y 5):")
print(
    ((df["LATITUD"] >= 4) & (df["LATITUD"] <= 5)).sum()
)

print("\nCantidad mayores a 5:")
print(
    (df["LATITUD"] > 5).sum()
)

# Probar división por 10^8
lat_8 = df["LATITUD"] / 100_000_000

print("\nDespués de dividir por 100.000.000:")
print(lat_8.describe())

print("\nValores entre 4 y 5 después de /10^8:")
print(
    ((lat_8 >= 4) & (lat_8 <= 5)).sum()
)

# Probar división por 10^9
lat_9 = df["LATITUD"] / 1_000_000_000

print("\nDespués de dividir por 1.000.000.000:")
print(lat_9.describe())

print("\nValores entre 4 y 5 después de /10^9:")
print(
    ((lat_9 >= 4) & (lat_9 <= 5)).sum()
)


# --------------------------------------------------
# LONGITUD
# --------------------------------------------------

print("\n===== LONGITUD =====")

print("\nValores originales:")
print(df["LONGITUD"].describe())

print("\nCantidad con longitud normal (entre -75 y -73):")
print(
    ((df["LONGITUD"] >= -75) & (df["LONGITUD"] <= -73)).sum()
)

# Probar división por 10^8
lon_8 = df["LONGITUD"] / 100_000_000

print("\nDespués de dividir por 100.000.000:")
print(lon_8.describe())

print("\nValores entre -75 y -73 después de /10^8:")
print(
    ((lon_8 >= -75) & (lon_8 <= -73)).sum()
)

# Probar división por 10^9
lon_9 = df["LONGITUD"] / 1_000_000_000

print("\nDespués de dividir por 1.000.000.000:")
print(lon_9.describe())

print("\nValores entre -75 y -73 después de /10^9:")
print(
    ((lon_9 >= -75) & (lon_9 <= -73)).sum()
)


# --------------------------------------------------
# X / Y
# --------------------------------------------------

print("\n===== X / Y =====")

print("\nX:")
print(df["X"].describe())

print("\nY:")
print(df["Y"].describe())

print("\nPrimeras 10 coordenadas originales:")

print(
    df[["X", "Y", "LATITUD", "LONGITUD"]]
    .head(10)
    .to_string(index=False)
)