"""Genera un mapa de revisión visual; no valida ni modifica candidatos."""

from html import escape
from pathlib import Path
import re

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import MarkerCluster


BASE = Path(__file__).resolve().parents[2]
ARCHIVO_REVISION = BASE / "models" / "hoja_revision_manual_georreferenciacion.csv"
ARCHIVO_DETALLE = BASE / "models" / "detalle_cruces_geometricos_50_grupos.csv"
ARCHIVO_LOCALIDADES = BASE / "data" / "raw" / "poligonos-localidad-2.geojson"
ARCHIVO_SALIDA = BASE / "models" / "mapa_revision_manual_georreferenciacion.html"

COLORES = [
    "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#66c2a5", "#e6ab02", "#666666",
]


def normalizar_direccion(valor):
    texto = re.sub(r"\s+", " ", str(valor).upper().strip())
    return re.sub(r"\s+0+(\d+)$", r" \1", texto)


def clave_ubicacion(codigo, direccion):
    return str(int(codigo)).zfill(2), normalizar_direccion(direccion)


def construir_mapa(revision, detalle, localidades):
    requeridas_revision = {
        "DIRECCION_NORMALIZADA", "CODIGO_LOCALIDAD", "NOMBRE_LOCALIDAD",
        "GRUPO_ESPACIAL", "LONGITUD_CALCULADA", "LATITUD_CALCULADA",
        "LONGITUD_ORIGINAL", "LATITUD_ORIGINAL", "CANTIDAD_CRUCES",
        "ESTADO_REVISION", "VALIDADO",
    }
    requeridas_detalle = {
        "DIRECCION", "CODIGO_LOCALIDAD", "GRUPO_ESPACIAL",
        "LONGITUD_CRUCE", "LATITUD_CRUCE", "DISTANCIA_METROS",
        "OBJECTID_VIA_1", "OBJECTID_VIA_2",
    }
    faltan = requeridas_revision - set(revision.columns)
    faltan |= requeridas_detalle - set(detalle.columns)
    if faltan:
        raise ValueError(f"Faltan columnas para el mapa: {sorted(faltan)}")

    if not revision["VALIDADO"].astype(str).str.upper().eq("NO").all():
        raise ValueError("El mapa de revisión solo admite candidatos no validados.")

    revision = revision.dropna(
        subset=["LONGITUD_CALCULADA", "LATITUD_CALCULADA"]
    ).copy()
    detalle = detalle.dropna(
        subset=["LONGITUD_CRUCE", "LATITUD_CRUCE"]
    ).copy()

    centro = [
        float(revision["LATITUD_CALCULADA"].mean()),
        float(revision["LONGITUD_CALCULADA"].mean()),
    ]
    mapa = folium.Map(
        location=centro,
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Mapa de calles",
        overlay=False,
        control=True,
    ).add_to(mapa)

    colores_localidad = {
        str(code).zfill(2): COLORES[i % len(COLORES)]
        for i, code in enumerate(
            sorted(localidades["identificador_unico_de_la_localidad"].astype(str))
        )
    }

    limites = folium.FeatureGroup(name="Límites de localidades", show=False)
    for _, poligono in localidades.iterrows():
        code = str(poligono["identificador_unico_de_la_localidad"]).zfill(2)
        folium.GeoJson(
            poligono.geometry.__geo_interface__,
            name=str(poligono["nombre_de_la_localidad"]),
            style_function=lambda _feature, color=colores_localidad.get(
                code, "#808080"
            ): {"color": color, "weight": 1, "fillOpacity": 0.02},
            tooltip=(
                f"{escape(str(poligono['nombre_de_la_localidad']))} "
                f"({escape(code)})"
            ),
        ).add_to(limites)
    limites.add_to(mapa)

    grupos_por_clave = {}
    for _, fila in revision.iterrows():
        grupos_por_clave.setdefault(
            clave_ubicacion(fila["CODIGO_LOCALIDAD"], fila["DIRECCION_NORMALIZADA"]),
            [],
        ).append(fila)

    candidatos = folium.FeatureGroup(
        name="Candidatos principales (NO validados)", show=True
    )
    originales = folium.FeatureGroup(
        name="Centros originales de los grupos", show=False
    )
    alternativos = folium.FeatureGroup(
        name="Cruces geométricos alternativos", show=True
    )
    cluster_alternativos = MarkerCluster(name="Alternativas agrupadas")

    for _, fila in revision.iterrows():
        code = str(int(fila["CODIGO_LOCALIDAD"])).zfill(2)
        direccion = escape(str(fila["DIRECCION_NORMALIZADA"]))
        localidad = escape(str(fila["NOMBRE_LOCALIDAD"]))
        grupo = escape(str(fila["GRUPO_ESPACIAL"]))
        estado = escape(str(fila["ESTADO_REVISION"]))
        llave = clave_ubicacion(fila["CODIGO_LOCALIDAD"], fila["DIRECCION_NORMALIZADA"])
        total_grupos = len(grupos_por_clave[llave])
        color = "#e67e22" if estado != "CANDIDATA_UNICA_PENDIENTE_VALIDACION" else "#2c7fb8"
        lat = float(fila["LATITUD_CALCULADA"])
        lon = float(fila["LONGITUD_CALCULADA"])

        contenido = (
            "<div style='width:300px'>"
            f"<b>{direccion}</b><br>{localidad} ({code})<hr>"
            f"Estado: {estado}<br>"
            f"Grupo espacial: {grupo}<br>"
            f"Cruces geométricos en este grupo: {int(fila['CANTIDAD_CRUCES'])}<br>"
            f"Grupos de esta dirección/localidad: {total_grupos}<br>"
            f"Distancia al cruce más cercano: {float(fila['DISTANCIA_METROS']):.2f} m<br>"
            f"Coordenada candidata: {lat:.6f}, {lon:.6f}<br>"
            "<b>Validado: NO</b></div>"
        )
        folium.CircleMarker(
            [lat, lon],
            radius=8,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"{direccion} — {localidad} — NO VALIDADO",
            popup=folium.Popup(contenido, max_width=340),
        ).add_to(candidatos)

        if pd.notna(fila["LONGITUD_ORIGINAL"]) and pd.notna(fila["LATITUD_ORIGINAL"]):
            folium.CircleMarker(
                [float(fila["LATITUD_ORIGINAL"]), float(fila["LONGITUD_ORIGINAL"])],
                radius=4,
                color="#222222",
                fill=True,
                fill_color="#ffffff",
                fill_opacity=1,
                tooltip=f"Centro original: {direccion}, grupo {grupo}",
            ).add_to(originales)

    for _, fila in detalle.iterrows():
        code = str(int(fila["CODIGO_LOCALIDAD"])).zfill(2)
        direccion = escape(normalizar_direccion(fila["DIRECCION"]))
        lat = float(fila["LATITUD_CRUCE"])
        lon = float(fila["LONGITUD_CRUCE"])
        popup = (
            "<div style='width:270px'>"
            f"<b>Cruce alternativo</b><br>{direccion}<br>Localidad: {code}<br>"
            f"Grupo: {escape(str(fila['GRUPO_ESPACIAL']))}<br>"
            f"Distancia al centro original: {float(fila['DISTANCIA_METROS']):.2f} m<br>"
            f"OBJECTID vía 1/2: {escape(str(fila['OBJECTID_VIA_1']))} / "
            f"{escape(str(fila['OBJECTID_VIA_2']))}<br><b>No validado</b></div>"
        )
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color="#555555",
            weight=1,
            fill=True,
            fill_color="#fdae61",
            fill_opacity=0.75,
            tooltip=f"Alternativa: {direccion} | {code}",
            popup=folium.Popup(popup, max_width=310),
        ).add_to(cluster_alternativos)

    cluster_alternativos.add_to(alternativos)
    candidatos.add_to(mapa)
    originales.add_to(mapa)
    alternativos.add_to(mapa)
    folium.LayerControl(collapsed=False).add_to(mapa)

    puntos = revision[["LATITUD_CALCULADA", "LONGITUD_CALCULADA"]].values.tolist()
    mapa.fit_bounds(puntos)
    return mapa


def main():
    for path in (ARCHIVO_REVISION, ARCHIVO_DETALLE, ARCHIVO_LOCALIDADES):
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")

    revision = pd.read_csv(ARCHIVO_REVISION, low_memory=False)
    detalle = pd.read_csv(ARCHIVO_DETALLE, low_memory=False)
    localidades = gpd.read_file(ARCHIVO_LOCALIDADES)
    if localidades.crs is None:
        raise ValueError("Los polígonos de localidades no tienen CRS definido.")
    if str(localidades.crs).upper() != "EPSG:4326":
        localidades = localidades.to_crs("EPSG:4326")

    mapa = construir_mapa(revision, detalle, localidades)
    mapa.save(str(ARCHIVO_SALIDA))
    print(f"Candidatos principales: {len(revision)}")
    print(f"Cruces geométricos alternativos: {len(detalle)}")
    print(f"Todos no validados: {revision['VALIDADO'].astype(str).str.upper().eq('NO').all()}")
    print(f"Mapa de revisión guardado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
