
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path

# =====================================================
# 1. RUTAS Y COORDENADA A INVESTIGAR
# =====================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO_MALLA = (
    BASE / "data" / "raw" /
    "Malla_Vial_Integral_Bogota_D_C.geojson"
)

SALIDA = (
    BASE / "models" /
    "auditoria_vial_av_sur_cl59s.csv"
)

# Coordenada candidata del grupo 0
LONGITUD = -74.177536
LATITUD = 4.596918

RADIO_METROS = 150

# =====================================================
# 2. CARGAR MALLA VIAL
# =====================================================

print("=" * 70)
print("AUDITORÍA DE INTERSECCIÓN SOBRE LA MALLA VIAL")
print("=" * 70)

print("\nCargando GeoJSON...")

vias = gpd.read_file(ARCHIVO_MALLA)

print("Elementos cargados:", len(vias))
print("Sistema de coordenadas:", vias.crs)

if vias.crs is None:
    raise ValueError("El GeoJSON no tiene CRS definido.")

# =====================================================
# 3. CREAR PUNTO DE REFERENCIA
# =====================================================

punto = gpd.GeoDataFrame(
    {
        "DIRECCION": ["AV AVENIDA DEL SUR - CL 59 S"],
        "LONGITUD": [LONGITUD],
        "LATITUD": [LATITUD]
    },
    geometry=[Point(LONGITUD, LATITUD)],
    crs="EPSG:4326"
)

# Proyectar para calcular distancias en metros
crs_metrico = vias.estimate_utm_crs()

if crs_metrico is None:
    raise ValueError("No fue posible determinar un CRS métrico.")

vias_m = vias.to_crs(crs_metrico)
punto_m = punto.to_crs(crs_metrico)

geom_punto = punto_m.geometry.iloc[0]

print("\nCoordenada investigada:")
print(f"Latitud: {LATITUD}")
print(f"Longitud: {LONGITUD}")
print("CRS métrico:", crs_metrico)

# =====================================================
# 4. BUSCAR VÍAS CERCANAS
# =====================================================

print(f"\nBuscando vías en un radio de {RADIO_METROS} metros...")

candidatas = vias_m[
    vias_m.geometry.notna()
    & vias_m.geometry.distance(geom_punto).le(RADIO_METROS)
].copy()

candidatas["DISTANCIA_METROS"] = (
    candidatas.geometry.distance(geom_punto)
)

candidatas = candidatas.sort_values(
    "DISTANCIA_METROS"
)

# =====================================================
# 5. MOSTRAR ATRIBUTOS IMPORTANTES
# =====================================================

campos = [
    "OBJECTID",
    "MVITIPO",
    "MVINANTIGU",
    "MVIETIQUET",
    "MVINOMBRE",
    "MVINALTERN",
    "NAME",
    "PREFIJO",
    "MVICODIGO",
    "MVI_VELREG",
    "DISTANCIA_METROS"
]

campos_disponibles = [
    campo for campo in campos
    if campo in candidatas.columns
]

resultado = candidatas[campos_disponibles].copy()

resultado.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 6. INFORME
# =====================================================

print("\n" + "=" * 70)
print("RESULTADOS DE LA AUDITORÍA")
print("=" * 70)

print("\nVías encontradas:", len(resultado))

if resultado.empty:
    print("No se encontraron vías dentro del radio investigado.")

else:
    print("\nVías más cercanas:")
    print(
        resultado.head(30).to_string(index=False)
    )

    print("\nNombres y etiquetas encontrados:")

    for campo in [
        "MVINANTIGU",
        "MVIETIQUET",
        "MVINOMBRE",
        "MVINALTERN",
        "NAME"
    ]:
        if campo in resultado.columns:
            valores = (
                resultado[campo]
                .dropna()
                .astype(str)
                .unique()
            )

            print(f"\n{campo}:")
            for valor in valores[:30]:
                if valor.strip():
                    print("-", valor)

print("\nArchivo generado:")
print(SALIDA)

print("\nAuditoría terminada.")
