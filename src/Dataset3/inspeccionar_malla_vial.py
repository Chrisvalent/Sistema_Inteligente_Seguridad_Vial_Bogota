
import geopandas as gpd
import os

RUTA = "data/raw/Malla_Vial_Integral_Bogota_D_C.geojson"

print("=" * 75)
print("INSPECCION DE LA MALLA VIAL DE BOGOTA")
print("=" * 75)

if not os.path.exists(RUTA):
    raise FileNotFoundError(f"No se encontró el archivo: {RUTA}")

print("\nLeyendo archivo geográfico...")
gdf = gpd.read_file(RUTA, engine="pyogrio")

print("\n1. INFORMACION GENERAL")
print("Registros:", len(gdf))
print("Columnas:", len(gdf.columns))
print("Sistema de coordenadas:", gdf.crs)
print("Geometrías:")
print(gdf.geometry.geom_type.value_counts().to_string())

print("\n2. CAMPOS DISPONIBLES")
for columna in gdf.columns:
    if columna != gdf.geometry.name:
        print("-", columna)

print("\n3. CAMPOS DE NOMENCLATURA")
campos_interes = [
    "MVITIPO",
    "MVINANTIGU",
    "MVIETIQUET",
    "MVINOMBRE",
    "MVINALTERN",
    "NAME",
    "MVI_VELREG",
    "MVIACAL"
]

for columna in campos_interes:
    if columna in gdf.columns:
        print(f"\n{columna}")
        print(
            gdf[columna]
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
            .to_string()
        )

print("\n4. MUESTRA DE ELEMENTOS")

columnas_muestra = [
    c for c in campos_interes
    if c in gdf.columns
]

print(gdf[columnas_muestra].head(15).to_string(index=False))

print("\n5. EXTENSION GEOGRAFICA")
print(gdf.total_bounds)

print("\n6. CALIDAD DE GEOMETRIAS")
print("Geometrías vacías:", int(gdf.geometry.is_empty.sum()))
print("Geometrías nulas:", int(gdf.geometry.isna().sum()))
print("Geometrías válidas:", int(gdf.geometry.is_valid.sum()))

print("\nInspección terminada.")
