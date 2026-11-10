
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
from html import escape

# ==================================================
# 1. RUTAS
# ==================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO = (
    BASE / "models" /
    "intersecciones_candidatos_normalizados.csv"
)

SALIDA = (
    BASE / "models" /
    "mapa_intersecciones_candidatas.html"
)

# ==================================================
# 2. CARGAR DATOS
# ==================================================

df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

columnas = [
    "DIRECCION_NORMALIZADA",
    "CODIGO_LOCALIDAD",
    "NOMBRE_LOCALIDAD",
    "GRUPO_ESPACIAL",
    "PUNTOS_ORIGINALES",
    "LONGITUD_CENTRO",
    "LATITUD_CENTRO",
    "VALIDADO"
]

faltantes = [c for c in columnas if c not in df.columns]

if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")

df = df.dropna(
    subset=["LONGITUD_CENTRO", "LATITUD_CENTRO"]
).copy()

df["LONGITUD_CENTRO"] = pd.to_numeric(
    df["LONGITUD_CENTRO"], errors="coerce"
)

df["LATITUD_CENTRO"] = pd.to_numeric(
    df["LATITUD_CENTRO"], errors="coerce"
)

df = df.dropna(
    subset=["LONGITUD_CENTRO", "LATITUD_CENTRO"]
)

# ==================================================
# 3. VALIDAR COORDENADAS
# ==================================================

df = df[
    df["LONGITUD_CENTRO"].between(-74.5, -73.8)
    & df["LATITUD_CENTRO"].between(3.7, 4.9)
].copy()

if df.empty:
    raise ValueError("No hay coordenadas válidas para mostrar.")

# ==================================================
# 4. CREAR MAPA
# ==================================================

centro_lat = df["LATITUD_CENTRO"].mean()
centro_lon = df["LONGITUD_CENTRO"].mean()

mapa = folium.Map(
    location=[centro_lat, centro_lon],
    zoom_start=11,
    tiles=None,
    control_scale=True
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/World_Street_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Tiles © Esri",
    name="Esri World Street Map",
    overlay=False,
    control=True
).add_to(mapa)

# Colores por localidad
colores = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
    "#fabebe", "#008080", "#e6beff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#000075",
    "#ffd8b1", "#dcbeff", "#fffac8", "#a9a9a9"
]

codigos = sorted(df["CODIGO_LOCALIDAD"].dropna().unique())
color_localidad = {
    codigo: colores[i % len(colores)]
    for i, codigo in enumerate(codigos)
}

# ==================================================
# 5. CREAR CAPAS POR LOCALIDAD
# ==================================================

for codigo in codigos:

    datos_localidad = df[
        df["CODIGO_LOCALIDAD"] == codigo
    ]

    nombre = str(
        datos_localidad["NOMBRE_LOCALIDAD"].iloc[0]
    )

    capa = folium.FeatureGroup(
        name=f"{nombre} ({codigo})",
        show=True
    )

    cluster = MarkerCluster(
        name=f"Candidatos {nombre}"
    )

    for _, fila in datos_localidad.iterrows():

        direccion = escape(
            str(fila["DIRECCION_NORMALIZADA"])
        )

        localidad = escape(
            str(fila["NOMBRE_LOCALIDAD"])
        )

        grupo = escape(str(fila["GRUPO_ESPACIAL"]))
        puntos = escape(str(fila["PUNTOS_ORIGINALES"]))
        validado = escape(str(fila["VALIDADO"]))

        lat = float(fila["LATITUD_CENTRO"])
        lon = float(fila["LONGITUD_CENTRO"])

        contenido = f"""
        <div style="width:260px">
            <h4>Intersección candidata</h4>
            <b>Dirección:</b> {direccion}<br>
            <b>Localidad:</b> {localidad}<br>
            <b>Código localidad:</b> {codigo}<br>
            <b>Grupo espacial:</b> {grupo}<br>
            <b>Puntos originales:</b> {puntos}<br>
            <b>Validación:</b> {validado}<br>
            <hr>
            <b>Latitud:</b> {lat:.6f}<br>
            <b>Longitud:</b> {lon:.6f}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color_localidad[codigo],
            weight=2,
            fill=True,
            fill_color=color_localidad[codigo],
            fill_opacity=0.85,
            tooltip=f"{direccion} | Grupo {grupo}",
            popup=folium.Popup(contenido, max_width=320)
        ).add_to(cluster)

    cluster.add_to(capa)
    capa.add_to(mapa)

# ==================================================
# 6. CONTROL Y AJUSTE DEL MAPA
# ==================================================

folium.LayerControl(
    collapsed=False
).add_to(mapa)

coordenadas = df[
    ["LATITUD_CENTRO", "LONGITUD_CENTRO"]
].values.tolist()

mapa.fit_bounds(coordenadas)

# ==================================================
# 7. GUARDAR MAPA
# ==================================================

mapa.save(str(SALIDA))

# ==================================================
# 8. RESUMEN
# ==================================================

print("=" * 65)
print("MAPA DE INTERSECCIONES CANDIDATAS")
print("=" * 65)

print(f"Coordenadas representadas: {len(df)}")
print(f"Localidades representadas: {len(codigos)}")
print(
    "Intersecciones diferentes:",
    df["DIRECCION_NORMALIZADA"].nunique()
)

print("\nCandidatos por localidad:")
print(
    df.groupby("NOMBRE_LOCALIDAD")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nMapa generado:")
print(SALIDA)

print("\nProceso terminado correctamente.")
