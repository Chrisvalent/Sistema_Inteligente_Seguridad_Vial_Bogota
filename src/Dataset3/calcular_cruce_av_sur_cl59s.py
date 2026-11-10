
import geopandas as gpd
import pandas as pd

from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import Transformer
from pathlib import Path

# =====================================================
# 1. CONFIGURACIÓN Y RUTAS
# =====================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO = (
    BASE / "data" / "raw" /
    "Malla_Vial_Integral_Bogota_D_C.geojson"
)

SALIDA = (
    BASE / "models" /
    "cruce_geometrico_av_sur_cl59s.csv"
)

# Coordenada candidata original
LONGITUD = -74.177536
LATITUD = 4.596918

RADIO_METROS = 150

# =====================================================
# 2. INICIO
# =====================================================

print("=" * 75)
print("CÁLCULO GEOMÉTRICO DE LA INTERSECCIÓN")
print("AVENIDA DEL SUR - CALLE 59 SUR")
print("=" * 75)

# =====================================================
# 3. CARGAR MALLA VIAL
# =====================================================

print("\nCargando malla vial...")

vias = gpd.read_file(ARCHIVO)

if vias.crs is None:
    raise ValueError(
        "La malla vial no tiene un sistema de coordenadas definido."
    )

print("Elementos cargados:", len(vias))
print("CRS original:", vias.crs)

# Proyección métrica
crs_metrico = vias.estimate_utm_crs()

if crs_metrico is None:
    raise ValueError(
        "No fue posible determinar el CRS métrico."
    )

vias = vias.to_crs(crs_metrico)

print("CRS métrico:", crs_metrico)

# Transformador de metros a coordenadas geográficas
transformador = Transformer.from_crs(
    crs_metrico,
    "EPSG:4326",
    always_xy=True
)

# =====================================================
# 4. CREAR PUNTO CANDIDATO
# =====================================================

punto_geografico = gpd.GeoSeries(
    [Point(LONGITUD, LATITUD)],
    crs="EPSG:4326"
)

punto = punto_geografico.to_crs(
    crs_metrico
).iloc[0]

print("\nCoordenada candidata:")
print("Latitud:", LATITUD)
print("Longitud:", LONGITUD)

# =====================================================
# 5. NORMALIZAR CAMPOS DE LA MALLA
# =====================================================

campos = [
    "MVINANTIGU",
    "MVIETIQUET",
    "MVINOMBRE",
    "MVINALTERN",
    "NAME"
]

for campo in campos:

    if campo not in vias.columns:
        vias[campo] = ""

    vias[campo] = (
        vias[campo]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

# =====================================================
# 6. IDENTIFICAR AVENIDA DEL SUR
# =====================================================

avenida = vias[
    (
        vias["MVINOMBRE"].str.contains(
            "AVENIDA DEL SUR",
            regex=False
        )
    )
    |
    (
        vias["MVINANTIGU"].eq("AK 78")
    )
    |
    (
        vias["MVIETIQUET"].eq("KR 77G")
    )
].copy()

# =====================================================
# 7. IDENTIFICAR CALLE 59 SUR
# =====================================================

calle = vias[
    (
        vias["MVINANTIGU"].eq("CL 59 S")
    )
    |
    (
        vias["MVIETIQUET"].eq("CL 59 S")
    )
].copy()

# =====================================================
# 8. FILTRAR POR PROXIMIDAD
# =====================================================

avenida = avenida[
    avenida.geometry.notna()
    & ~avenida.geometry.is_empty
    & avenida.geometry.distance(punto).le(RADIO_METROS)
].copy()

calle = calle[
    calle.geometry.notna()
    & ~calle.geometry.is_empty
    & calle.geometry.distance(punto).le(RADIO_METROS)
].copy()

print("\nSegmentos de Avenida del Sur:", len(avenida))
print("Segmentos de Calle 59 Sur:", len(calle))

if avenida.empty or calle.empty:
    raise ValueError(
        "No se encontraron segmentos suficientes de ambas vías."
    )

# =====================================================
# 9. CALCULAR INTERSECCIONES GEOMÉTRICAS
# =====================================================

print("\nCalculando intersecciones geométricas...")

resultados = []

for idx_a, fila_a in avenida.iterrows():

    for idx_c, fila_c in calle.iterrows():

        geom_a = fila_a.geometry
        geom_c = fila_c.geometry

        if not geom_a.intersects(geom_c):
            continue

        cruce = geom_a.intersection(geom_c)

        if cruce.is_empty:
            continue

        # Una intersección de líneas puede producir puntos,
        # multipuntos o geometrías mixtas.
        def extraer_puntos(geometria):

            if geometria.is_empty:
                return []

            if geometria.geom_type == "Point":
                return [geometria]

            if geometria.geom_type in (
                "MultiPoint",
                "GeometryCollection"
            ):
                puntos = []

                for elemento in geometria.geoms:
                    puntos.extend(
                        extraer_puntos(elemento)
                    )

                return puntos

            return []

        puntos_cruce = extraer_puntos(cruce)

        for punto_cruce in puntos_cruce:

            distancia = punto.distance(punto_cruce)

            if distancia > RADIO_METROS:
                continue

            # Convertir coordenadas métricas a geográficas
            longitud, latitud = transformador.transform(
                punto_cruce.x,
                punto_cruce.y
            )

            resultados.append({

                "OBJECTID_AVENIDA": fila_a.get(
                    "OBJECTID"
                ),

                "OBJECTID_CALLE": fila_c.get(
                    "OBJECTID"
                ),

                "NOMBRE_AVENIDA": fila_a.get(
                    "MVINOMBRE"
                ),

                "NOMBRE_CALLE": fila_c.get(
                    "MVINANTIGU"
                ),

                "LONGITUD_CRUCE": longitud,

                "LATITUD_CRUCE": latitud,

                "DISTANCIA_CANDIDATO_METROS": distancia

            })

# =====================================================
# 10. CONSOLIDAR RESULTADOS
# =====================================================

resultado = pd.DataFrame(resultados)

if resultado.empty:

    print("\nNo se encontraron cruces geométricos.")
    print("La coordenada permanece sin validar.")

else:

    resultado = resultado.drop_duplicates(
        subset=[
            "LONGITUD_CRUCE",
            "LATITUD_CRUCE"
        ]
    )

    resultado = resultado.sort_values(
        "DISTANCIA_CANDIDATO_METROS"
    ).reset_index(drop=True)

    resultado.insert(
        0,
        "ID_CRUCE",
        range(1, len(resultado) + 1)
    )

    # =================================================
    # 11. CLASIFICAR DISTANCIA
    # =================================================

    def clasificar_distancia(distancia):

        if distancia <= 10:
            return "COINCIDENCIA MUY ALTA"

        elif distancia <= 25:
            return "COINCIDENCIA ALTA"

        elif distancia <= 50:
            return "REVISAR"

        return "DISTANCIA ALTA"

    resultado["CLASIFICACION"] = (
        resultado["DISTANCIA_CANDIDATO_METROS"]
        .apply(clasificar_distancia)
    )

    # =================================================
    # 12. GUARDAR CSV
    # =================================================

    resultado.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    # =================================================
    # 13. MOSTRAR RESULTADOS
    # =================================================

    print("\n" + "=" * 75)
    print("RESULTADOS DEL CÁLCULO GEOMÉTRICO")
    print("=" * 75)

    print(
        "\nCruces geométricos encontrados:",
        len(resultado)
    )

    print("\nCruces ordenados por distancia:")

    print(
        resultado.to_string(index=False)
    )

    mejor = resultado.iloc[0]

    print("\n" + "=" * 75)
    print("MEJOR CANDIDATO GEOMÉTRICO")
    print("=" * 75)

    print(
        "Latitud:",
        round(mejor["LATITUD_CRUCE"], 8)
    )

    print(
        "Longitud:",
        round(mejor["LONGITUD_CRUCE"], 8)
    )

    print(
        "Distancia al candidato:",
        round(
            mejor["DISTANCIA_CANDIDATO_METROS"],
            2
        ),
        "metros"
    )

    print(
        "Clasificación:",
        mejor["CLASIFICACION"]
    )

    print("\nArchivo generado:")
    print(SALIDA)

# =====================================================
# 14. FINALIZACIÓN
# =====================================================

print("\n" + "=" * 75)
print("PROCESO TERMINADO")
print("=" * 75)
