import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_ARCHIVO = "data/raw/siniestros_viales_consolidados_bogota_dc.xlsx"

HOJAS = [
    "SINIESTROS",
    "ACTOR_VIAL",
    "VEHICULOS",
    "HIPOTESIS",
    "DICCIONARIO"
]


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("DIAGNÓSTICO - SINIESTROS VIALES CONSOLIDADOS")
print("=" * 70)


# ============================================================
# CARGAR ARCHIVO
# ============================================================

print("\nCargando archivo...")

archivo = pd.ExcelFile(RUTA_ARCHIVO)

print("\nHojas encontradas:")

for hoja in archivo.sheet_names:
    print("-", hoja)


# ============================================================
# DIAGNÓSTICO DE CADA HOJA
# ============================================================

datos = {}

for hoja in HOJAS:

    print("\n" + "=" * 70)
    print(f"HOJA: {hoja}")
    print("=" * 70)

    df = pd.read_excel(
        RUTA_ARCHIVO,
        sheet_name=hoja
    )

    datos[hoja] = df

    print(f"\nRegistros: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    print("\nColumnas:")

    for columna in df.columns:
        print("-", columna)

    print("\nPrimeros registros:")

    print(df.head(3).to_string(index=False))

    print("\nValores faltantes:")

    faltantes = df.isnull().sum()

    faltantes = faltantes[faltantes > 0]

    if len(faltantes) == 0:
        print("No hay valores faltantes.")
    else:
        print(faltantes)


# ============================================================
# ANÁLISIS DE CODIGO_ACCIDENTE
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS DE CODIGO_ACCIDENTE")
print("=" * 70)


for hoja in [
    "SINIESTROS",
    "ACTOR_VIAL",
    "VEHICULOS",
    "HIPOTESIS"
]:

    df = datos[hoja]

    print(f"\n--- {hoja} ---")

    total = len(df)

    unicos = df["CODIGO_ACCIDENTE"].nunique()

    duplicados = df["CODIGO_ACCIDENTE"].duplicated().sum()

    print(f"Registros:              {total:,}")
    print(f"Códigos únicos:         {unicos:,}")
    print(f"Registros repetidos:    {duplicados:,}")


# ============================================================
# COBERTURA ENTRE TABLAS
# ============================================================

print("\n" + "=" * 70)
print("COBERTURA ENTRE TABLAS")
print("=" * 70)


codigos_siniestros = set(
    datos["SINIESTROS"]["CODIGO_ACCIDENTE"].dropna()
)

print(
    f"\nAccidentes únicos en SINIESTROS: "
    f"{len(codigos_siniestros):,}"
)


for hoja in [
    "ACTOR_VIAL",
    "VEHICULOS",
    "HIPOTESIS"
]:

    codigos = set(
        datos[hoja]["CODIGO_ACCIDENTE"].dropna()
    )

    interseccion = codigos_siniestros.intersection(codigos)

    faltan = codigos_siniestros - codigos

    porcentaje = (
        len(interseccion) /
        len(codigos_siniestros) *
        100
    )

    print(f"\n{hoja}")

    print(
        f"Accidentes con información: "
        f"{len(interseccion):,}"
    )

    print(
        f"Accidentes sin información:  "
        f"{len(faltan):,}"
    )

    print(
        f"Cobertura:                   "
        f"{porcentaje:.2f}%"
    )


# ============================================================
# RELACIÓN 1:N
# ============================================================

print("\n" + "=" * 70)
print("RELACIÓN DE REGISTROS POR ACCIDENTE")
print("=" * 70)


for hoja in [
    "ACTOR_VIAL",
    "VEHICULOS",
    "HIPOTESIS"
]:

    df = datos[hoja]

    conteo = (
        df.groupby("CODIGO_ACCIDENTE")
        .size()
    )

    print(f"\n--- {hoja} ---")

    print(
        f"Promedio de registros por accidente: "
        f"{conteo.mean():.2f}"
    )

    print(
        f"Máximo de registros de un accidente: "
        f"{conteo.max()}"
    )

    print(
        f"Accidentes con más de un registro: "
        f"{(conteo > 1).sum():,}"
    )


# ============================================================
# GRAVEDAD
# ============================================================

print("\n" + "=" * 70)
print("GRAVEDAD EN SINIESTROS")
print("=" * 70)

df_siniestros = datos["SINIESTROS"]

print(
    df_siniestros["GRAVEDAD"]
    .value_counts(dropna=False)
)


# ============================================================
# DICCIONARIO
# ============================================================

print("\n" + "=" * 70)
print("DICCIONARIO DE DATOS")
print("=" * 70)

diccionario = datos["DICCIONARIO"]

print(
    f"\nRegistros del diccionario: "
    f"{len(diccionario):,}"
)

print("\nCampos disponibles:")

print(diccionario.columns.tolist())

print("\nPrimeros registros:")

print(
    diccionario.head(20)
    .to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO FINALIZADO")
print("=" * 70)