import pandas as pd

ruta = "data/raw/Historico_siniestros_bogota.xlsx"

df = pd.read_excel(ruta)


# ==========================================================
# FUNCIONES DE NORMALIZACIÓN
# ==========================================================

def normalizar_latitud(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)
    absoluto = abs(valor)

    # Coordenada ya normalizada
    if 4 <= absoluto <= 5:
        return valor

    # Escala x 1.000
    elif 1_000 <= absoluto < 10_000:
        return valor / 1_000

    # Escala x 100.000.000
    elif 100_000_000 <= absoluto < 1_000_000_000:
        return valor / 100_000_000

    # Escala x 10.000.000.000.000
    elif 10_000_000_000_000 <= absoluto < 100_000_000_000_000:
        return valor / 10_000_000_000_000

    else:
        return None


def normalizar_longitud(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)
    absoluto = abs(valor)

    # Coordenada ya normalizada
    if 73 <= absoluto <= 75:
        return valor

    # Escala x 1.000
    elif 70_000 <= absoluto < 100_000:
        return valor / 1_000

    # Escala x 100.000.000
    elif 7_000_000_000 <= absoluto < 8_000_000_000:
        return valor / 100_000_000

    # Escala x 10.000.000.000.000
    elif 700_000_000_000_000 <= absoluto < 800_000_000_000_000:
        return valor / 10_000_000_000_000

    else:
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
print("VALIDACIÓN DE COORDENADAS NORMALIZADAS")
print("=" * 60)

print("\n===== LATITUD NORMALIZADA =====")

print(df["LATITUD_NORMALIZADA"].describe())

print("\nValores no convertidos:")
print(df["LATITUD_NORMALIZADA"].isna().sum())

print("\nValores fuera del rango esperado:")

lat_fuera = df[
    ~df["LATITUD_NORMALIZADA"].between(4, 5)
    & df["LATITUD_NORMALIZADA"].notna()
]

print(len(lat_fuera))


print("\n===== LONGITUD NORMALIZADA =====")

print(df["LONGITUD_NORMALIZADA"].describe())

print("\nValores no convertidos:")
print(df["LONGITUD_NORMALIZADA"].isna().sum())

print("\nValores fuera del rango esperado:")

lon_fuera = df[
    ~df["LONGITUD_NORMALIZADA"].between(-75, -73)
    & df["LONGITUD_NORMALIZADA"].notna()
]

print(len(lon_fuera))


# ==========================================================
# EJEMPLOS
# ==========================================================

print("\n===== EJEMPLOS DE CONVERSIÓN =====")

print(
    df[
        [
            "LATITUD",
            "LATITUD_NORMALIZADA",
            "LONGITUD",
            "LONGITUD_NORMALIZADA"
        ]
    ].head(20).to_string(index=False)
)