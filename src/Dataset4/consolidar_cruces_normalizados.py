
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# ==========================================================
# CONFIGURACION
# ==========================================================

BASE = Path(__file__).resolve().parents[2]

ENTRADA = (
    BASE / "models" / "detalle_cruces_geometricos_50_grupos.csv"
)

SALIDA_DETALLE = (
    BASE / "models" / "auditoria_cruces_normalizados.csv"
)

SALIDA_RESUMEN = (
    BASE / "models" / "resumen_cruces_normalizados.csv"
)

RADIO_AGRUPACION_METROS = 20
RADIO_TIERRA_METROS = 6371000

# ==========================================================
# NORMALIZACION
# ==========================================================

def normalizar_direccion(valor):
    if pd.isna(valor):
        return ""

    texto = str(valor).upper().strip()
    texto = re.sub(r"\s+", " ", texto)

    # Normalizar terminaciones 02, 002, etc. a 2
    texto = re.sub(r"\s+0+(\d+)$", r" \1", texto)

    return texto


def distancia_haversine(lat1, lon1, lat2, lon2):
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

    return (
        2
        * RADIO_TIERRA_METROS
        * math.asin(math.sqrt(a))
    )


# ==========================================================
# LECTURA
# ==========================================================

print("\n==============================================")
print(" CONSOLIDACION DE CRUCES NORMALIZADOS")
print("==============================================")

if not ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo:\n{ENTRADA}"
    )

df = pd.read_csv(
    ENTRADA,
    encoding="utf-8-sig"
)

print(f"\nRegistros originales: {len(df)}")

# ==========================================================
# VALIDAR COLUMNAS
# ==========================================================

requeridas = [
    "DIRECCION",
    "CODIGO_LOCALIDAD",
    "NOMBRE_LOCALIDAD",
    "GRUPO_ESPACIAL",
    "VIA_1",
    "VIA_2",
    "OBJECTID_VIA_1",
    "OBJECTID_VIA_2",
    "LONGITUD_CRUCE",
    "LATITUD_CRUCE",
    "DISTANCIA_METROS"
]

faltantes = [
    c for c in requeridas
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas necesarias: {faltantes}"
    )

# ==========================================================
# CONVERSION Y LIMPIEZA
# ==========================================================

for columna in [
    "LONGITUD_CRUCE",
    "LATITUD_CRUCE",
    "DISTANCIA_METROS"
]:
    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )

df["CODIGO_LOCALIDAD"] = (
    df["CODIGO_LOCALIDAD"].astype(str).str.strip()
)

df["DIRECCION_NORMALIZADA"] = (
    df["DIRECCION"].apply(normalizar_direccion)
)

df["CLAVE_DIRECCION"] = (
    df["CODIGO_LOCALIDAD"]
    + "|"
    + df["DIRECCION_NORMALIZADA"]
)

# Eliminar coordenadas no válidas del análisis espacial,
# conservando las filas en el detalle.
coordenadas_validas = (
    df["LATITUD_CRUCE"].between(-90, 90)
    & df["LONGITUD_CRUCE"].between(-180, 180)
    & df["LATITUD_CRUCE"].notna()
    & df["LONGITUD_CRUCE"].notna()
)

df["COORDENADA_VALIDA"] = coordenadas_validas

df["CLUSTER_ESPACIAL"] = pd.NA

# ==========================================================
# AGRUPAR PUNTOS CERCANOS POR DIRECCION Y LOCALIDAD
# ==========================================================

resumenes = []

for clave, grupo in df.groupby(
    "CLAVE_DIRECCION",
    sort=True
):

    indices_validos = grupo.index[
        grupo["COORDENADA_VALIDA"]
    ].tolist()

    etiquetas = {}

    if indices_validos:

        coordenadas = np.radians(
            df.loc[
                indices_validos,
                ["LATITUD_CRUCE", "LONGITUD_CRUCE"]
            ].to_numpy(dtype=float)
        )

        modelo = DBSCAN(
            eps=RADIO_AGRUPACION_METROS
            / RADIO_TIERRA_METROS,
            min_samples=1,
            metric="haversine",
            algorithm="ball_tree"
        )

        etiquetas = modelo.fit_predict(coordenadas)

        for indice, etiqueta in zip(
            indices_validos,
            etiquetas
        ):
            df.at[
                indice,
                "CLUSTER_ESPACIAL"
            ] = int(etiqueta)

    grupo_completo = df.loc[grupo.index]

    cantidad_clusters = (
        grupo_completo.loc[
            grupo_completo["COORDENADA_VALIDA"],
            "CLUSTER_ESPACIAL"
        ]
        .nunique()
    )

    cantidad_grupos = (
        grupo_completo["GRUPO_ESPACIAL"]
        .nunique()
    )

    cantidad_puntos = len(grupo_completo)

    cantidad_objectids = (
        grupo_completo[
            ["OBJECTID_VIA_1", "OBJECTID_VIA_2"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    distancias = grupo_completo[
        "DISTANCIA_METROS"
    ].dropna()

    distancia_minima = (
        round(distancias.min(), 2)
        if not distancias.empty
        else np.nan
    )

    distancia_maxima = (
        round(distancias.max(), 2)
        if not distancias.empty
        else np.nan
    )

    resumenes.append({
        "CODIGO_LOCALIDAD": grupo_completo[
            "CODIGO_LOCALIDAD"
        ].iloc[0],

        "NOMBRE_LOCALIDAD": grupo_completo[
            "NOMBRE_LOCALIDAD"
        ].iloc[0],

        "DIRECCION_NORMALIZADA": grupo_completo[
            "DIRECCION_NORMALIZADA"
        ].iloc[0],

        "VARIANTES_DIRECCION": " || ".join(
            sorted(
                grupo_completo["DIRECCION"]
                .dropna()
                .astype(str)
                .unique()
            )
        ),

        "CANTIDAD_REGISTROS_CRUCE": cantidad_puntos,

        "CANTIDAD_GRUPOS_ORIGINALES": cantidad_grupos,

        "PARES_OBJECTID_DISTINTOS": cantidad_objectids,

        "UBICACIONES_ESPACIALES_20M": int(
            cantidad_clusters
        ),

        "DISTANCIA_MINIMA_ORIGEN_CRUCE_M": distancia_minima,

        "DISTANCIA_MAXIMA_ORIGEN_CRUCE_M": distancia_maxima
    })

# ==========================================================
# EXPORTAR
# ==========================================================

resumen = pd.DataFrame(resumenes)

df.to_csv(
    SALIDA_DETALLE,
    index=False,
    encoding="utf-8-sig"
)

resumen = resumen.sort_values(
    [
        "UBICACIONES_ESPACIALES_20M",
        "CANTIDAD_REGISTROS_CRUCE"
    ],
    ascending=[False, False]
)

resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# INFORME
# ==========================================================

print("\n==============================================")
print(" RESULTADOS")
print("==============================================")

print(f"\nCruces analizados: {len(df)}")

print(
    "Direcciones originales distintas:",
    df["DIRECCION"].nunique()
)

print(
    "Direcciones normalizadas por localidad:",
    len(resumen)
)

print(
    "Registros con coordenadas válidas:",
    int(df["COORDENADA_VALIDA"].sum())
)

print("\nDistribución de ubicaciones espaciales:")

print(
    resumen["UBICACIONES_ESPACIALES_20M"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nDIRECCIONES CON MAS UBICACIONES ESPACIALES:")

print(
    resumen[
        [
            "NOMBRE_LOCALIDAD",
            "DIRECCION_NORMALIZADA",
            "CANTIDAD_REGISTROS_CRUCE",
            "CANTIDAD_GRUPOS_ORIGINALES",
            "UBICACIONES_ESPACIALES_20M"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nArchivos generados:")
print(SALIDA_DETALLE)
print(SALIDA_RESUMEN)

print("\nNOTA:")
print(
    "La agrupacion de 20 metros es exploratoria. "
    "No constituye validacion de una direccion ni "
    "seleccion automatica de coordenadas."
)

print("\nProceso terminado.")
