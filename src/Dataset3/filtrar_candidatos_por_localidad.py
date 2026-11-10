
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# RUTAS
# ============================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO_LOCALIDADES = (
    BASE / "data" / "raw" / "poligonos-localidad-2.geojson"
)

ARCHIVO_CANDIDATOS = (
    BASE / "models" / "candidatos_intersecciones.csv"
)

ARCHIVO_RESUMEN = (
    BASE / "models" / "resumen_georreferenciacion.csv"
)

SALIDA_CANDIDATOS = (
    BASE / "models" / "candidatos_filtrados_localidad.csv"
)

SALIDA_RESUMEN = (
    BASE / "models" / "resumen_filtrado_localidad.csv"
)


# ============================================================
# NORMALIZAR CÓDIGOS DE LOCALIDAD
# ============================================================

def normalizar_codigo(valor):

    if pd.isna(valor):
        return ""

    try:
        return str(int(float(valor))).zfill(2)
    except (ValueError, TypeError):
        return str(valor).strip().zfill(2)


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("VALIDACIÓN ESPACIAL POR LOCALIDAD")
    print("=" * 70)

    for archivo in [
        ARCHIVO_LOCALIDADES,
        ARCHIVO_CANDIDATOS,
        ARCHIVO_RESUMEN
    ]:

        if not archivo.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {archivo}"
            )

    # --------------------------------------------------------
    # Cargar polígonos
    # --------------------------------------------------------

    print("\nCargando polígonos de localidades...")

    localidades = gpd.read_file(ARCHIVO_LOCALIDADES)

    if localidades.crs is None:
        localidades = localidades.set_crs("EPSG:4326")
    else:
        localidades = localidades.to_crs("EPSG:4326")

    campo_codigo = "identificador_unico_de_la_localidad"
    campo_nombre = "nombre_de_la_localidad"

    for campo in [campo_codigo, campo_nombre]:

        if campo not in localidades.columns:
            raise ValueError(
                f"No existe el campo {campo} en los polígonos."
            )

    localidades["CODIGO_LOCALIDAD_NORM"] = (
        localidades[campo_codigo].apply(normalizar_codigo)
    )

    localidades["NOMBRE_LOCALIDAD"] = localidades[campo_nombre]

    print(
        f"Polígonos cargados: {len(localidades)}"
    )

    # --------------------------------------------------------
    # Cargar candidatos y resumen original
    # --------------------------------------------------------

    candidatos = pd.read_csv(
        ARCHIVO_CANDIDATOS,
        encoding="utf-8-sig"
    )

    resumen = pd.read_csv(
        ARCHIVO_RESUMEN,
        encoding="utf-8-sig"
    )

    print(
        f"Candidatos geométricos originales: {len(candidatos)}"
    )

    # --------------------------------------------------------
    # Normalizar códigos
    # --------------------------------------------------------

    candidatos["CODIGO_LOCALIDAD_NORM"] = (
        candidatos["CODIGO_LOCALIDAD"].apply(normalizar_codigo)
    )

    resumen["CODIGO_LOCALIDAD_NORM"] = (
        resumen["CODIGO_LOCALIDAD"].apply(normalizar_codigo)
    )

    # --------------------------------------------------------
    # Crear geometrías de los candidatos
    # --------------------------------------------------------

    candidatos["geometry"] = candidatos.apply(
        lambda fila: Point(
            float(fila["LONGITUD"]),
            float(fila["LATITUD"])
        ),
        axis=1
    )

    puntos = gpd.GeoDataFrame(
        candidatos,
        geometry="geometry",
        crs="EPSG:4326"
    )

    # --------------------------------------------------------
    # Comparar cada candidato con su localidad
    # --------------------------------------------------------

    resultados = []

    print("\nEvaluando candidatos contra los polígonos...")

    for _, fila in puntos.iterrows():

        codigo = fila["CODIGO_LOCALIDAD_NORM"]

        punto = fila.geometry

        poligono = localidades[
            localidades["CODIGO_LOCALIDAD_NORM"] == codigo
        ]

        dentro = False
        nombre_localidad = ""

        if len(poligono) == 1:

            geometria_localidad = poligono.iloc[0].geometry

            nombre_localidad = (
                poligono.iloc[0]["NOMBRE_LOCALIDAD"]
            )

            dentro = geometria_localidad.covers(punto)

        registro = fila.drop(labels=["geometry"]).to_dict()

        registro["NOMBRE_LOCALIDAD"] = nombre_localidad

        registro["DENTRO_LOCALIDAD"] = dentro

        registro["VALIDACION_LOCALIDAD"] = (
            "COMPATIBLE" if dentro else "FUERA DE LOCALIDAD"
        )

        resultados.append(registro)

    df_validacion = pd.DataFrame(resultados)

    # --------------------------------------------------------
    # Conservar únicamente candidatos compatibles
    # --------------------------------------------------------

    df_compatibles = df_validacion[
        df_validacion["DENTRO_LOCALIDAD"] == True
    ].copy()

    # --------------------------------------------------------
    # Contar candidatos compatibles por dirección
    # --------------------------------------------------------

    conteos = (
        df_compatibles.groupby("DIRECCION")
        .size()
        .to_dict()
    )

    resumen["CANDIDATOS_EN_LOCALIDAD"] = (
        resumen["DIRECCION"].map(conteos).fillna(0).astype(int)
    )

    def clasificar(cantidad):

        if cantidad == 0:
            return "SIN CANDIDATO EN LOCALIDAD"

        if cantidad == 1:
            return "UN CANDIDATO COMPATIBLE"

        return "VARIOS CANDIDATOS COMPATIBLES"

    resumen["ESTADO_LOCALIDAD"] = (
        resumen["CANDIDATOS_EN_LOCALIDAD"].apply(clasificar)
    )

    # --------------------------------------------------------
    # Exportar
    # --------------------------------------------------------

    df_validacion.to_csv(
        SALIDA_CANDIDATOS,
        index=False,
        encoding="utf-8-sig"
    )

    resumen.to_csv(
        SALIDA_RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTADOS DE VALIDACIÓN ESPACIAL")
    print("=" * 70)

    print("\nCandidatos originales:", len(df_validacion))

    print(
        "Candidatos dentro de la localidad:",
        len(df_compatibles)
    )

    print(
        "Candidatos fuera de la localidad:",
        len(df_validacion) - len(df_compatibles)
    )

    print("\nEstado por dirección:")

    print(
        resumen["ESTADO_LOCALIDAD"]
        .value_counts()
        .to_string()
    )

    print("\nPrimeras 20 direcciones:")

    print(
        resumen[
            [
                "DIRECCION",
                "CODIGO_LOCALIDAD",
                "CANDIDATOS",
                "CANDIDATOS_EN_LOCALIDAD",
                "ESTADO_LOCALIDAD"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nArchivos generados:")

    print(SALIDA_CANDIDATOS)

    print(SALIDA_RESUMEN)

    print("\nProceso terminado.")

    print(
        "\nIMPORTANTE: compatible con localidad no significa "
        "coordenada definitivamente validada."
    )


if __name__ == "__main__":
    main()
