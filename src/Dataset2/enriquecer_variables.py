import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARCHIVO_EXCEL = (
    BASE_DIR
    / "data"
    / "raw"
    / "siniestros_viales_consolidados_bogota_dc.xlsx"
)

ARCHIVO_INTEGRADO = (
    BASE_DIR
    / "data"
    / "processed"
    / "siniestros_integrados.csv"
)

ARCHIVO_SALIDA = (
    BASE_DIR
    / "data"
    / "processed"
    / "siniestros_enriquecido.csv"
)


print("=" * 70)
print("ENRIQUECIMIENTO DEL DATASET INTEGRADO")
print("=" * 70)


# ============================================================
# CARGAR INFORMACIÓN
# ============================================================

print("\nCargando dataset integrado...")

df = pd.read_csv(ARCHIVO_INTEGRADO)

print(f"Accidentes iniciales: {len(df):,}")


print("\nCargando tablas originales...")

vehiculos = pd.read_excel(
    ARCHIVO_EXCEL,
    sheet_name="VEHICULOS"
)

hipotesis = pd.read_excel(
    ARCHIVO_EXCEL,
    sheet_name="HIPOTESIS"
)

diccionario = pd.read_excel(
    ARCHIVO_EXCEL,
    sheet_name="DICCIONARIO"
)


# ============================================================
# 1. VEHÍCULOS
# ============================================================

print("\n" + "=" * 70)
print("1. ENRIQUECIMIENTO DE VEHÍCULOS")
print("=" * 70)


# Diccionario de clases
dic_vehiculos = diccionario[
    (diccionario["HOJA"] == "VEHICULOS") &
    (diccionario["CAMPO"] == "CLASE")
].copy()

mapa_clases = dict(
    zip(
        dic_vehiculos["CODIGO"],
        dic_vehiculos["DESCRIPCION"]
    )
)

print(f"Clases de vehículo en diccionario: {len(mapa_clases)}")


# Crear columnas por clase
for codigo, descripcion in mapa_clases.items():

    nombre = (
        "NUM_"
        + str(descripcion)
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(",", "")
    )

    # Evitar nombres demasiado problemáticos
    nombre = (
        nombre
        .replace("__", "_")
        .replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
    )

    conteos = (
        vehiculos.assign(
            MARCA_CLASE=(
                vehiculos["CLASE"] == codigo
            ).astype(int)
        )
        .groupby("CODIGO_ACCIDENTE")["MARCA_CLASE"]
        .sum()
        .rename(nombre)
    )

    df = df.merge(
        conteos,
        on="CODIGO_ACCIDENTE",
        how="left"
    )


# Motocicletas: agrupar clases relacionadas
columnas_moto = [
    c for c in df.columns
    if c in [
        "NUM_MOTOCICLETA",
        "NUM_MOTOCICLO",
        "NUM_CUATRIMOTO"
    ]
]

if columnas_moto:

    df["NUM_MOTOS_TOTAL"] = (
        df[columnas_moto]
        .fillna(0)
        .sum(axis=1)
    )


# ============================================================
# 2. HIPÓTESIS / CAUSAS
# ============================================================

print("\n" + "=" * 70)
print("2. ENRIQUECIMIENTO DE HIPÓTESIS")
print("=" * 70)


dic_causas = diccionario[
    (diccionario["HOJA"] == "HIPOTESIS") &
    (diccionario["CAMPO"] == "CODIGO_CAUSA")
].copy()

mapa_causas = dict(
    zip(
        dic_causas["CODIGO"],
        dic_causas["DESCRIPCION"]
    )
)

print(f"Causas disponibles en diccionario: {len(mapa_causas)}")


# Crear variables binarias solamente para causas
# suficientemente representadas en los datos.

frecuencias = hipotesis["CODIGO_CAUSA"].value_counts()

MIN_CASOS = 100

causas_seleccionadas = [
    codigo
    for codigo in mapa_causas
    if frecuencias.get(codigo, 0) >= MIN_CASOS
]

print(
    f"Causas con al menos {MIN_CASOS} registros: "
    f"{len(causas_seleccionadas)}"
)


for codigo in causas_seleccionadas:

    descripcion = mapa_causas[codigo]

    nombre = (
        "CAUSA_"
        + str(codigo)
        + "_"
        + str(descripcion)
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(",", "")
        .replace(".", "")
        .replace("/", "_")
    )

    nombre = (
        nombre
        .replace("__", "_")
        .replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
    )

    accidentes_con_causa = set(
        hipotesis.loc[
            hipotesis["CODIGO_CAUSA"] == codigo,
            "CODIGO_ACCIDENTE"
        ]
    )

    df[nombre] = (
        df["CODIGO_ACCIDENTE"]
        .isin(accidentes_con_causa)
        .astype(int)
    )


# ============================================================
# 3. VARIABLES AGRUPADAS DE CAUSAS
# ============================================================

print("\n" + "=" * 70)
print("3. VARIABLES AGRUPADAS DE CAUSAS")
print("=" * 70)


def crear_grupo_causa(nombre, codigos):

    existentes = [
        codigo
        for codigo in codigos
        if codigo in mapa_causas
    ]

    if not existentes:
        return

    accidentes = set(
        hipotesis.loc[
            hipotesis["CODIGO_CAUSA"].isin(existentes),
            "CODIGO_ACCIDENTE"
        ]
    )

    df[nombre] = (
        df["CODIGO_ACCIDENTE"]
        .isin(accidentes)
        .astype(int)
    )


# Conductor
crear_grupo_causa(
    "CAUSA_CONDUCTOR",
    range(101, 158)
)

# Vehículo
crear_grupo_causa(
    "CAUSA_VEHICULO",
    range(201, 218)
)

# Vía
crear_grupo_causa(
    "CAUSA_VIA",
    range(301, 309)
)

# Peatón
crear_grupo_causa(
    "CAUSA_PEATON",
    range(401, 412)
)

# Pasajero / acompañante
crear_grupo_causa(
    "CAUSA_PASAJERO",
    range(501, 507)
)

# Ciclista / motociclista
crear_grupo_causa(
    "CAUSA_CICLISTA_MOTOCICLISTA",
    range(90, 100)
)


# ============================================================
# 4. VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("4. VALIDACIÓN")
print("=" * 70)

print(f"\nRegistros finales: {len(df):,}")
print(f"Columnas finales: {len(df.columns):,}")

duplicados = df["CODIGO_ACCIDENTE"].duplicated().sum()

print(f"Duplicados CODIGO_ACCIDENTE: {duplicados:,}")


# ============================================================
# 5. GUARDAR
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO DATASET ENRIQUECIDO")
print("=" * 70)


df.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nArchivo generado:")
print(ARCHIVO_SALIDA)

print("\n" + "=" * 70)
print("ENRIQUECIMIENTO FINALIZADO")
print("=" * 70)