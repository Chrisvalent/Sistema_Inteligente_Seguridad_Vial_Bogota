import pandas as pd

ruta = "data/raw/Historico_siniestros_bogota.xlsx"

df = pd.read_excel(ruta)


# ==========================================================
# NORMALIZAR LATITUD
# ==========================================================

def normalizar_latitud(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)

    # Ya está normalizada
    if 4 <= abs(valor) <= 5:
        return valor

    # Mover el punto decimal hacia la izquierda
    while abs(valor) > 5:
        valor = valor / 10

    # Validar resultado
    if 4 <= abs(valor) <= 5:
        return valor

    return None


# ==========================================================
# NORMALIZAR LONGITUD
# ==========================================================

def normalizar_longitud(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)

    # Ya está normalizada
    if 73 <= abs(valor) <= 75:
        return valor

    # Mover el punto decimal hacia la izquierda
    while abs(valor) > 75:
        valor = valor / 10

    # Validar resultado
    if 73 <= abs(valor) <= 75:
        return valor

    return None


# ==========================================================
# APLICAR NORMALIZACIÓN
# ==========================================================

df["LATITUD_NORMALIZADA"] = df["LATITUD"].apply(
    normalizar_latitud
)

df["LONGITUD_NORMALIZADA"] = df["LONGITUD"].apply(
    normalizar_longitud
)


# ==========================================================
# RESULTADOS
# ==========================================================

print("=" * 60)
print("VALIDACIÓN DEFINITIVA DE COORDENADAS")
print("=" * 60)

print("\n===== LATITUD =====")

print("Total:", len(df))

print(
    "Convertidas:",
    df["LATITUD_NORMALIZADA"].notna().sum()
)

print(
    "No convertidas:",
    df["LATITUD_NORMALIZADA"].isna().sum()
)

print(
    "Fuera de rango:",
    (
        ~df["LATITUD_NORMALIZADA"].between(4, 5)
        & df["LATITUD_NORMALIZADA"].notna()
    ).sum()
)

print("\nEstadísticas:")
print(df["LATITUD_NORMALIZADA"].describe())


print("\n===== LONGITUD =====")

print(
    "Convertidas:",
    df["LONGITUD_NORMALIZADA"].notna().sum()
)

print(
    "No convertidas:",
    df["LONGITUD_NORMALIZADA"].isna().sum()
)

print(
    "Fuera de rango:",
    (
        ~df["LONGITUD_NORMALIZADA"].between(-75, -73)
        & df["LONGITUD_NORMALIZADA"].notna()
    ).sum()
)

print("\nEstadísticas:")
print(df["LONGITUD_NORMALIZADA"].describe())


# ==========================================================
# EJEMPLOS ESPECÍFICOS
# ==========================================================

print("\n===== EJEMPLOS =====")

ejemplos = df[
    ["LATITUD", "LATITUD_NORMALIZADA",
     "LONGITUD", "LONGITUD_NORMALIZADA"]
].head(20)

print(ejemplos.to_string(index=False))