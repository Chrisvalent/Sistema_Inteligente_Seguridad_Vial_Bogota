
from pathlib import Path
import re
import math
import pandas as pd

# ==========================================================
# CONFIGURACION
# ==========================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO_ENTRADA = (
    BASE / "models" / "auditoria_georreferenciacion_50_grupos.csv"
)

ARCHIVO_SALIDA = (
    BASE / "models" / "revision_calidad_georreferenciacion.csv"
)

# ==========================================================
# FUNCIONES
# ==========================================================

def normalizar_direccion(valor):
    if pd.isna(valor):
        return ""

    texto = str(valor).upper().strip()
    texto = re.sub(r"\s+", " ", texto)

    # Normalizar terminaciones numericas como 02 y 2
    texto = re.sub(r"\s+0+(\d+)$", r" \1", texto)

    return texto


def distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Distancia aproximada entre dos coordenadas en metros.
    """

    radio = 6371000

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radio * c


def obtener_columna(df, posibles):
    """
    Encuentra una columna aunque cambie ligeramente su nombre.
    """

    columnas = {
        str(col).strip().upper(): col
        for col in df.columns
    }

    for nombre in posibles:
        if nombre.upper() in columnas:
            return columnas[nombre.upper()]

    return None


# ==========================================================
# LECTURA
# ==========================================================

print("\n==============================================")
print(" AUDITORIA DE GEORREFERENCIACION")
print("==============================================")

if not ARCHIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo:\n{ARCHIVO_ENTRADA}"
    )

df = pd.read_csv(
    ARCHIVO_ENTRADA,
    encoding="utf-8-sig"
)

df = df.reset_index(drop=True)

print(f"\nRegistros leidos: {len(df)}")
print("\nColumnas encontradas:")
print(df.columns.tolist())

# ==========================================================
# IDENTIFICAR COLUMNAS
# ==========================================================

col_direccion = obtener_columna(
    df, ["DIRECCION"]
)

col_localidad = obtener_columna(
    df, ["CODIGO_LOCALIDAD", "NOMBRE_LOCALIDAD"]
)

col_grupo = obtener_columna(
    df, ["GRUPO_ESPACIAL"]
)

col_cruces = obtener_columna(
    df, ["CANTIDAD_CRUCES", "NUMERO_CRUCES"]
)

col_distancia = obtener_columna(
    df, ["DISTANCIA_METROS", "DISTANCIA_CRUCE_METROS"]
)

col_lat = obtener_columna(
    df, ["LATITUD_CALCULADA", "LATITUD"]
)

col_lon = obtener_columna(
    df, ["LONGITUD_CALCULADA", "LONGITUD"]
)

requeridas = {
    "DIRECCION": col_direccion,
    "CANTIDAD_CRUCES": col_cruces,
    "DISTANCIA_METROS": col_distancia,
    "LATITUD_CALCULADA": col_lat,
    "LONGITUD_CALCULADA": col_lon
}

faltantes = [
    nombre
    for nombre, columna in requeridas.items()
    if columna is None
]

if faltantes:
    print("\nNo fue posible identificar estas columnas:")
    for nombre in faltantes:
        print(f" - {nombre}")

    raise ValueError(
        "Revisa los nombres de las columnas impresos anteriormente. "
        "No se modifico el archivo original."
    )

# ==========================================================
# NORMALIZACION
# ==========================================================

df["DIRECCION_NORMALIZADA"] = (
    df[col_direccion].apply(normalizar_direccion)
)

df["_CRUCES"] = pd.to_numeric(
    df[col_cruces],
    errors="coerce"
)

df["_DISTANCIA"] = pd.to_numeric(
    df[col_distancia],
    errors="coerce"
)

df["_LAT"] = pd.to_numeric(
    df[col_lat],
    errors="coerce"
)

df["_LON"] = pd.to_numeric(
    df[col_lon],
    errors="coerce"
)

# ==========================================================
# CLASIFICACION DE CALIDAD
# ==========================================================

def clasificar_fila(fila):

    cruces = fila["_CRUCES"]
    distancia = fila["_DISTANCIA"]
    lat = fila["_LAT"]
    lon = fila["_LON"]

    if pd.isna(lat) or pd.isna(lon):
        return "SIN_COORDENADAS"

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "COORDENADAS_INVALIDAS"

    if pd.isna(cruces) or cruces == 0:
        return "SIN_CRUCES"

    if pd.isna(distancia):
        return "DISTANCIA_NO_DISPONIBLE"

    if distancia > 100:
        return "REVISAR_DISTANCIA"

    if cruces > 1:
        return "REVISAR_MULTIPLES_CRUCES"

    if distancia <= 10:
        return "CANDIDATA_PRIORITARIA"

    return "REVISAR_COINCIDENCIA"


df["REVISION_TECNICA"] = df.apply(
    clasificar_fila,
    axis=1
)

# ==========================================================
# NUMERO DE GRUPOS POR DIRECCION
# ==========================================================

claves = ["DIRECCION_NORMALIZADA"]

if col_localidad is not None:
    claves.append(col_localidad)

df["TOTAL_GRUPOS_DIRECCION"] = (
    df.groupby(claves)["DIRECCION_NORMALIZADA"]
    .transform("size")
)

# ==========================================================
# DISTANCIA ENTRE GRUPOS DE UNA MISMA DIRECCION
# ==========================================================

df["DISTANCIA_ENTRE_GRUPOS_METROS"] = pd.NA

for _, grupo in df.groupby(claves, dropna=False):

    indices = grupo.index.tolist()

    if len(indices) < 2:
        continue

    for indice in indices:

        lat1 = df.at[indice, "_LAT"]
        lon1 = df.at[indice, "_LON"]

        if pd.isna(lat1) or pd.isna(lon1):
            continue

        distancias = []

        for otro in indices:

            if otro == indice:
                continue

            lat2 = df.at[otro, "_LAT"]
            lon2 = df.at[otro, "_LON"]

            if pd.isna(lat2) or pd.isna(lon2):
                continue

            distancias.append(
                distancia_haversine(
                    lat1, lon1, lat2, lon2
                )
            )

        if distancias:
            df.at[
                indice,
                "DISTANCIA_ENTRE_GRUPOS_METROS"
            ] = round(min(distancias), 2)

# ==========================================================
# RESULTADO
# ==========================================================

columnas_auxiliares = [
    "_CRUCES",
    "_DISTANCIA",
    "_LAT",
    "_LON"
]

df.drop(
    columns=columnas_auxiliares,
    inplace=True
)

df.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\n==============================================")
print(" RESULTADOS DE LA AUDITORIA")
print("==============================================")

print("\nClasificacion tecnica:")

print(
    df["REVISION_TECNICA"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nCantidad de grupos por direccion:")

print(
    df["TOTAL_GRUPOS_DIRECCION"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nPrimeros 15 registros:")

columnas_mostrar = [
    col_direccion,
    "REVISION_TECNICA",
    "TOTAL_GRUPOS_DIRECCION",
    "DISTANCIA_ENTRE_GRUPOS_METROS"
]

print(
    df[columnas_mostrar]
    .head(15)
    .to_string(index=False)
)

print("\nArchivo generado:")
print(ARCHIVO_SALIDA)

print("\nIMPORTANTE:")
print(
    "Esta auditoria clasifica candidatos tecnicos. "
    "No declara coordenadas como geograficamente validadas."
)

print("\nProceso terminado.")
