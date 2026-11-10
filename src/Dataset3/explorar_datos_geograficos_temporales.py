
from pathlib import Path
import re
import pandas as pd


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

RAIZ = Path(__file__).resolve().parents[2]

RUTA_CSV = (
    RAIZ / "data" / "processed" / "siniestros_modelo.csv"
)

CARPETA_SALIDA = RAIZ / "models"

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Variables temporales y de análisis conocidas
VARIABLES_ANALISIS = [
    "ANO",
    "MES",
    "DIA",
    "DIA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "FRANJA_HORARIA",
    "HORA_NUM",
    "CODIGO_LOCALIDAD",
    "GRAVEDAD",
]

# Palabras que pueden identificar información geográfica
PATRON_GEO = re.compile(
    r"LAT|LONG|COORD|LOCALIDAD|LOCALIZ|DIRECCION|"
    r"BARRIO|UPZ|SECTOR|INTERSECCION|TRAMO|VIA|"
    r"MANZANA|GEOMET|SHAPE|POINT",
    re.IGNORECASE,
)


# ==========================================================
# FUNCIONES
# ==========================================================

def encabezado(texto):
    print("\n" + "=" * 65)
    print(texto)
    print("=" * 65)


def guardar_csv(df, nombre):
    ruta = CARPETA_SALIDA / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"Archivo generado: {ruta}")


# ==========================================================
# PROCESO PRINCIPAL
# ==========================================================

def main():

    encabezado("ANÁLISIS GEOGRÁFICO Y TEMPORAL DE SINIESTROS")

    if not RUTA_CSV.exists():
        print("ERROR: No se encontró el archivo:")
        print(RUTA_CSV)
        return

    print("Leyendo dataset...")

    df = pd.read_csv(RUTA_CSV)

    print(f"Registros: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    # ------------------------------------------------------
    # 1. INVENTARIO DE COLUMNAS
    # ------------------------------------------------------

    encabezado("1. INVENTARIO DE VARIABLES")

    inventario = pd.DataFrame({
        "variable": df.columns,
        "tipo_dato": df.dtypes.astype(str).values,
        "valores_no_nulos": df.notna().sum().values,
        "valores_nulos": df.isna().sum().values,
        "porcentaje_nulos": (
            df.isna().mean().values * 100
        ).round(2),
    })

    guardar_csv(
        inventario,
        "inventario_columnas_geograficas.csv"
    )

    columnas_geo = [
        columna
        for columna in df.columns
        if PATRON_GEO.search(columna)
    ]

    print("\nPosibles variables geográficas:")

    if columnas_geo:
        for columna in columnas_geo:
            porcentaje = df[columna].isna().mean() * 100

            print(
                f"- {columna}: "
                f"{porcentaje:.2f}% de valores faltantes"
            )
    else:
        print("No se identificaron columnas geográficas por nombre.")

    print("\nEjemplos de valores geográficos:")

    for columna in columnas_geo:
        ejemplos = (
            df[columna]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        print(f"{columna}: {ejemplos}")

    # ------------------------------------------------------
    # 2. VARIABLES TEMPORALES
    # ------------------------------------------------------

    encabezado("2. DISPONIBILIDAD DE VARIABLES TEMPORALES")

    for columna in VARIABLES_ANALISIS:
        if columna in df.columns:
            print(
                f"{columna}: "
                f"{df[columna].notna().sum():,} valores disponibles"
            )
        else:
            print(f"{columna}: NO EXISTE")

    # ------------------------------------------------------
    # 3. SINIESTROS POR LOCALIDAD
    # ------------------------------------------------------

    encabezado("3. SINIESTROS POR LOCALIDAD")

    if "CODIGO_LOCALIDAD" in df.columns:

        localidad = (
            df.groupby("CODIGO_LOCALIDAD", dropna=False)
            .size()
            .reset_index(name="cantidad_siniestros")
            .sort_values(
                "cantidad_siniestros",
                ascending=False
            )
        )

        localidad["porcentaje"] = (
            localidad["cantidad_siniestros"]
            / len(df) * 100
        ).round(2)

        guardar_csv(
            localidad,
            "resumen_siniestros_localidad.csv"
        )

        print("\nPrimeras 10 localidades:")

        print(localidad.head(10).to_string(index=False))

    else:
        print("No existe CODIGO_LOCALIDAD.")

    # ------------------------------------------------------
    # 4. SINIESTROS POR GRAVEDAD
    # ------------------------------------------------------

    encabezado("4. DISTRIBUCIÓN POR GRAVEDAD")

    if "GRAVEDAD" in df.columns:

        gravedad = (
            df.groupby("GRAVEDAD", dropna=False)
            .size()
            .reset_index(name="cantidad")
            .sort_values("cantidad", ascending=False)
        )

        gravedad["porcentaje"] = (
            gravedad["cantidad"] / len(df) * 100
        ).round(2)

        guardar_csv(
            gravedad,
            "resumen_siniestros_gravedad.csv"
        )

        print(gravedad.to_string(index=False))

    else:
        print("No existe la columna GRAVEDAD.")

    # ------------------------------------------------------
    # 5. SINIESTROS POR HORA
    # ------------------------------------------------------

    encabezado("5. DISTRIBUCIÓN POR HORA")

    if "HORA_NUM" in df.columns:

        df["HORA_ANALISIS"] = (
            pd.to_numeric(df["HORA_NUM"], errors="coerce")
            .apply(
                lambda x: int(x) % 24
                if pd.notna(x) else pd.NA
            )
        )

        por_hora = (
            df.groupby("HORA_ANALISIS", dropna=False)
            .size()
            .reset_index(name="cantidad_siniestros")
            .sort_values("HORA_ANALISIS")
        )

        guardar_csv(
            por_hora,
            "resumen_siniestros_hora.csv"
        )

        print(
            por_hora.sort_values(
                "cantidad_siniestros",
                ascending=False
            ).head(10).to_string(index=False)
        )

    else:
        print("No existe HORA_NUM.")

    # ------------------------------------------------------
    # 6. SINIESTROS POR DÍA DE LA SEMANA
    # ------------------------------------------------------

    encabezado("6. DISTRIBUCIÓN POR DÍA DE LA SEMANA")

    if "DIA_SEMANA" in df.columns:

        por_dia = (
            df.groupby("DIA_SEMANA", dropna=False)
            .size()
            .reset_index(name="cantidad_siniestros")
            .sort_values(
                "cantidad_siniestros",
                ascending=False
            )
        )

        guardar_csv(
            por_dia,
            "resumen_siniestros_dia_semana.csv"
        )

        print(por_dia.to_string(index=False))

    else:
        print("No existe DIA_SEMANA.")

    # ------------------------------------------------------
    # 7. MAPA DE CALOR DÍA Y HORA
    # ------------------------------------------------------

    encabezado("7. MATRIZ DÍA Y HORA")

    if {"DIA_SEMANA", "HORA_NUM"}.issubset(df.columns):

        df["HORA_ANALISIS"] = (
            pd.to_numeric(df["HORA_NUM"], errors="coerce")
            .apply(
                lambda x: int(x) % 24
                if pd.notna(x) else pd.NA
            )
        )

        matriz = pd.pivot_table(
            df,
            index="DIA_SEMANA",
            columns="HORA_ANALISIS",
            values="GRAVEDAD" if "GRAVEDAD" in df.columns else "ANO",
            aggfunc="count",
            fill_value=0,
        )

        guardar_csv(
            matriz.reset_index(),
            "matriz_siniestros_dia_hora.csv"
        )

        print(matriz.to_string())

    else:
        print("Faltan variables para construir la matriz.")

    # ------------------------------------------------------
    # 8. LOCALIDAD Y GRAVEDAD
    # ------------------------------------------------------

    encabezado("8. GRAVEDAD POR LOCALIDAD")

    if {
        "CODIGO_LOCALIDAD",
        "GRAVEDAD"
    }.issubset(df.columns):

        matriz_localidad = pd.crosstab(
            df["CODIGO_LOCALIDAD"],
            df["GRAVEDAD"],
            dropna=False
        ).reset_index()

        guardar_csv(
            matriz_localidad,
            "matriz_localidad_gravedad.csv"
        )

        print(matriz_localidad.head(10).to_string(index=False))

    # ------------------------------------------------------
    # 9. EVALUACIÓN FINAL DE DISPONIBILIDAD GEOGRÁFICA
    # ------------------------------------------------------

    encabezado("9. CONCLUSIÓN DE DISPONIBILIDAD DE DATOS")

    palabras_lat = [
        c for c in df.columns
        if re.search(r"LATITUD|LATITUDE|LAT$", c, re.I)
    ]

    palabras_lon = [
        c for c in df.columns
        if re.search(r"LONGITUD|LONGITUDE|LON$", c, re.I)
    ]

    print("Posibles columnas de latitud:", palabras_lat)
    print("Posibles columnas de longitud:", palabras_lon)

    if palabras_lat and palabras_lon:

        lat = pd.to_numeric(
            df[palabras_lat[0]], errors="coerce"
        )
        lon = pd.to_numeric(
            df[palabras_lon[0]], errors="coerce"
        )

        coordenadas_validas = (
            lat.between(-90, 90)
            & lon.between(-180, 180)
        )

        print(
            "Registros con ambas coordenadas válidas:",
            f"{coordenadas_validas.sum():,}"
        )

    else:
        print(
            "No se identificó un par claro de columnas "
            "de latitud y longitud."
        )

    print("\nANÁLISIS FINALIZADO.")
    print("Revisa los archivos generados en la carpeta models.")


if __name__ == "__main__":
    main()
