import pandas as pd

ruta = "data/raw/Historico_siniestros_bogota.xlsx"

df = pd.read_excel(ruta)

print("=" * 60)
print("INVESTIGACIÓN DE COORDENADAS NO CONVERTIDAS")
print("=" * 60)


# ==========================================================
# LATITUDES NO CONVERTIDAS
# ==========================================================

def latitud_normalizada(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)
    a = abs(valor)

    if 4 <= a <= 5:
        return valor
    elif 1_000 <= a < 10_000:
        return valor / 1_000
    elif 100_000_000 <= a < 1_000_000_000:
        return valor / 100_000_000
    elif 10_000_000_000_000 <= a < 100_000_000_000_000:
        return valor / 10_000_000_000_000

    return None


def longitud_normalizada(valor):

    if pd.isna(valor):
        return None

    valor = float(valor)
    a = abs(valor)

    if 73 <= a <= 75:
        return valor
    elif 70_000 <= a < 100_000:
        return valor / 1_000
    elif 7_000_000_000 <= a < 8_000_000_000:
        return valor / 100_000_000
    elif 700_000_000_000_000 <= a < 800_000_000_000_000:
        return valor / 10_000_000_000_000

    return None


df["LAT_TEST"] = df["LATITUD"].apply(latitud_normalizada)
df["LON_TEST"] = df["LONGITUD"].apply(longitud_normalizada)


# ==========================================================
# VALORES QUE FALTAN
# ==========================================================

lat_no_convertidas = df[df["LAT_TEST"].isna()]["LATITUD"]

lon_no_convertidas = df[df["LON_TEST"].isna()]["LONGITUD"]


print("\n===== LATITUDES NO CONVERTIDAS =====")

print("Cantidad:", len(lat_no_convertidas))

print("\nPrimeros 30 valores:")

print(
    lat_no_convertidas
    .head(30)
    .to_string(index=False)
)


print("\n===== LONGITUDES NO CONVERTIDAS =====")

print("Cantidad:", len(lon_no_convertidas))

print("\nPrimeros 30 valores:")

print(
    lon_no_convertidas
    .head(30)
    .to_string(index=False)
)


# ==========================================================
# MAGNITUDES
# ==========================================================

print("\n===== MAGNITUDES DE LATITUD NO CONVERTIDA =====")

print(
    lat_no_convertidas
    .abs()
    .apply(lambda x: len(str(int(x))))
    .value_counts()
    .sort_index()
)


print("\n===== MAGNITUDES DE LONGITUD NO CONVERTIDA =====")

print(
    lon_no_convertidas
    .abs()
    .apply(lambda x: len(str(int(x))))
    .value_counts()
    .sort_index()
)