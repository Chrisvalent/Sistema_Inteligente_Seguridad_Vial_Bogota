
import geopandas as gpd
import pandas as pd
import re
import unicodedata

from shapely.geometry import Point
from pyproj import Transformer
from pathlib import Path

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO_CANDIDATOS = (
    BASE / "models" /
    "candidatos_agrupados_espacialmente.csv"
)

ARCHIVO_MALLA = (
    BASE / "data" / "raw" /
    "Malla_Vial_Integral_Bogota_D_C.geojson"
)

SALIDA_RESUMEN = (
    BASE / "models" /
    "auditoria_georreferenciacion_50_grupos.csv"
)

SALIDA_DETALLE = (
    BASE / "models" /
    "detalle_cruces_geometricos_50_grupos.csv"
)

RADIO_METROS = 150

# =========================================================
# 2. FUNCIONES DE NORMALIZACIÓN
# =========================================================

def normalizar(valor):

    if pd.isna(valor):
        return ""

    texto = str(valor).upper().strip()

    texto = unicodedata.normalize(
        "NFKD", texto
    ).encode("ASCII", "ignore").decode("ASCII")

    texto = re.sub(r"[^A-Z0-9 ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def alias_via(valor):

    """
    Genera formas alternativas de identificación
    para las vías de nomenclatura bogotana.
    """

    texto = normalizar(valor)

    if not texto:
        return set()

    alias = {texto}

    # Vías tipo KR, CL, AK, AC, TV y DG
    patron = re.match(
        r"^(KR|CL|AK|AC|TV|DG)\s*"
        r"(\d+[A-Z]?)\s*(S|N)?$",
        texto
    )

    if patron:

        tipo, numero, orientacion = patron.groups()

        orientacion = orientacion or ""

        if tipo == "KR":
            alias.add(f"AK {numero}{' ' + orientacion if orientacion else ''}")

        elif tipo == "AK":
            alias.add(f"KR {numero}{' ' + orientacion if orientacion else ''}")

        alias.add(
            f"{tipo} {numero}{' ' + orientacion if orientacion else ''}"
        )

    # Algunas avenidas aparecen con o sin prefijo AV
    if texto.startswith("AVENIDA "):
        alias.add(texto.replace("AVENIDA ", "", 1))

    elif texto.startswith("AV "):
        alias.add(texto.replace("AV ", "", 1))

    return alias


# =========================================================
# 3. IDENTIFICAR SI UN SEGMENTO PERTENECE A UNA VÍA
# =========================================================

def coincide_via(fila, aliases, es_numerica=False):

    campos = [
        "MVINANTIGU",
        "MVIETIQUET",
        "MVINOMBRE",
        "MVINALTERN",
        "NAME"
    ]

    valores = {
        campo: normalizar(fila.get(campo, ""))
        for campo in campos
    }

    for alias in aliases:

        for campo, valor in valores.items():

            if not valor:
                continue

            # Para nomenclatura numérica se exige
            # coincidencia exacta.
            if es_numerica:

                if valor == alias:
                    return True

            else:

                # Para avenidas con nombre se admite
                # coincidencia del nombre dentro del atributo.
                if valor == alias or alias in valor:
                    return True

    return False


def es_nomenclatura_numerica(valor):

    texto = normalizar(valor)

    return bool(
        re.match(
            r"^(KR|CL|AK|AC|TV|DG)\s*\d+",
            texto
        )
    )


# =========================================================
# 4. EXTRAER PUNTOS DE UNA INTERSECCIÓN GEOMÉTRICA
# =========================================================

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
            puntos.extend(extraer_puntos(elemento))

        return puntos

    return []


# =========================================================
# 5. CARGAR CANDIDATOS
# =========================================================

print("=" * 75)
print("GEORREFERENCIACIÓN AUTOMATIZADA DE INTERSECCIONES")
print("=" * 75)

print("\nCargando candidatos...")

df = pd.read_csv(
    ARCHIVO_CANDIDATOS,
    encoding="utf-8-sig"
)

columnas_requeridas = [
    "DIRECCION",
    "CODIGO_LOCALIDAD",
    "NOMBRE_LOCALIDAD",
    "VIA_1",
    "VIA_2",
    "GRUPO_ESPACIAL",
    "PUNTOS_ORIGINALES",
    "LONGITUD_CENTRO",
    "LATITUD_CENTRO"
]

faltantes = [
    c for c in columnas_requeridas
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas en candidatos: {faltantes}"
    )

df = df.dropna(
    subset=[
        "LONGITUD_CENTRO",
        "LATITUD_CENTRO",
        "VIA_1",
        "VIA_2"
    ]
).copy()

df["LONGITUD_CENTRO"] = pd.to_numeric(
    df["LONGITUD_CENTRO"],
    errors="coerce"
)

df["LATITUD_CENTRO"] = pd.to_numeric(
    df["LATITUD_CENTRO"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "LONGITUD_CENTRO",
        "LATITUD_CENTRO"
    ]
)

print("Grupos candidatos:", len(df))

# =========================================================
# 6. CARGAR MALLA VIAL
# =========================================================

print("\nCargando malla vial...")

vias = gpd.read_file(ARCHIVO_MALLA)

if vias.crs is None:
    raise ValueError(
        "La malla vial no tiene CRS definido."
    )

crs_metrico = vias.estimate_utm_crs()

if crs_metrico is None:
    raise ValueError(
        "No se pudo determinar el CRS métrico."
    )

vias = vias.to_crs(crs_metrico)

print("Segmentos de la malla:", len(vias))
print("CRS métrico:", crs_metrico)

# Asegurar geometrías utilizables
vias = vias[
    vias.geometry.notna()
    & ~vias.geometry.is_empty
].copy()

for campo in [
    "MVINANTIGU",
    "MVIETIQUET",
    "MVINOMBRE",
    "MVINALTERN",
    "NAME"
]:

    if campo not in vias.columns:
        vias[campo] = ""

# Transformador de coordenadas
transformador = Transformer.from_crs(
    crs_metrico,
    "EPSG:4326",
    always_xy=True
)

# =========================================================
# 7. PROCESAR CADA GRUPO ESPACIAL
# =========================================================

resultados_resumen = []
resultados_detalle = []

print("\nIniciando análisis de intersecciones...\n")

for numero, (_, fila) in enumerate(
    df.iterrows(),
    start=1
):

    direccion = str(fila["DIRECCION"])

    localidad = fila["NOMBRE_LOCALIDAD"]

    via_1 = str(fila["VIA_1"])
    via_2 = str(fila["VIA_2"])

    grupo = fila["GRUPO_ESPACIAL"]

    longitud_original = float(
        fila["LONGITUD_CENTRO"]
    )

    latitud_original = float(
        fila["LATITUD_CENTRO"]
    )

    print(
        f"[{numero}/{len(df)}] "
        f"{direccion} | Grupo {grupo}"
    )

    # -----------------------------------------------------
    # 7.1 Coordenada original
    # -----------------------------------------------------

    punto_geo = gpd.GeoSeries(
        [
            Point(
                longitud_original,
                latitud_original
            )
        ],
        crs="EPSG:4326"
    )

    punto = punto_geo.to_crs(
        crs_metrico
    ).iloc[0]

    # -----------------------------------------------------
    # 7.2 Identificar las vías
    # -----------------------------------------------------

    aliases_1 = alias_via(via_1)
    aliases_2 = alias_via(via_2)

    numerica_1 = es_nomenclatura_numerica(via_1)
    numerica_2 = es_nomenclatura_numerica(via_2)

    mascara_1 = vias.apply(
        lambda f: coincide_via(
            f,
            aliases_1,
            numerica_1
        ),
        axis=1
    )

    mascara_2 = vias.apply(
        lambda f: coincide_via(
            f,
            aliases_2,
            numerica_2
        ),
        axis=1
    )

    segmentos_1 = vias[mascara_1].copy()
    segmentos_2 = vias[mascara_2].copy()

    # -----------------------------------------------------
    # 7.3 Filtrar por proximidad
    # -----------------------------------------------------

    segmentos_1 = segmentos_1[
        segmentos_1.geometry.distance(punto)
        <= RADIO_METROS
    ]

    segmentos_2 = segmentos_2[
        segmentos_2.geometry.distance(punto)
        <= RADIO_METROS
    ]

    cruces = []

    # -----------------------------------------------------
    # 7.4 Calcular cruces
    # -----------------------------------------------------

    for _, segmento_1 in segmentos_1.iterrows():

        for _, segmento_2 in segmentos_2.iterrows():

            geom_1 = segmento_1.geometry
            geom_2 = segmento_2.geometry

            if not geom_1.intersects(geom_2):
                continue

            interseccion = geom_1.intersection(
                geom_2
            )

            puntos = extraer_puntos(
                interseccion
            )

            for punto_cruce in puntos:

                distancia = punto.distance(
                    punto_cruce
                )

                longitud, latitud = (
                    transformador.transform(
                        punto_cruce.x,
                        punto_cruce.y
                    )
                )

                cruce = {

                    "DIRECCION": direccion,

                    "CODIGO_LOCALIDAD": fila[
                        "CODIGO_LOCALIDAD"
                    ],

                    "NOMBRE_LOCALIDAD": localidad,

                    "GRUPO_ESPACIAL": grupo,

                    "VIA_1": via_1,

                    "VIA_2": via_2,

                    "OBJECTID_VIA_1": segmento_1.get(
                        "OBJECTID"
                    ),

                    "OBJECTID_VIA_2": segmento_2.get(
                        "OBJECTID"
                    ),

                    "LONGITUD_ORIGINAL": longitud_original,

                    "LATITUD_ORIGINAL": latitud_original,

                    "LONGITUD_CRUCE": longitud,

                    "LATITUD_CRUCE": latitud,

                    "DISTANCIA_METROS": distancia

                }

                cruces.append(cruce)

    # -----------------------------------------------------
    # 7.5 Consolidar cruces encontrados
    # -----------------------------------------------------

    cruces_df = pd.DataFrame(cruces)

    if not cruces_df.empty:

        cruces_df = cruces_df.drop_duplicates(
            subset=[
                "LONGITUD_CRUCE",
                "LATITUD_CRUCE"
            ]
        )

        cruces_df = cruces_df.sort_values(
            "DISTANCIA_METROS"
        ).reset_index(drop=True)

        for _, cruce in cruces_df.iterrows():

            resultados_detalle.append(
                cruce.to_dict()
            )

        mejor = cruces_df.iloc[0]

        distancia = float(
            mejor["DISTANCIA_METROS"]
        )

        if distancia <= 10:
            estado = "COINCIDENCIA MUY ALTA"

        elif distancia <= 25:
            estado = "COINCIDENCIA ALTA"

        elif distancia <= 50:
            estado = "REVISAR"

        else:
            estado = "DISTANCIA ALTA"

        if len(cruces_df) > 1:
            estado += " - MULTIPLES CRUCES"

        resultados_resumen.append({

            "DIRECCION": direccion,

            "CODIGO_LOCALIDAD": fila[
                "CODIGO_LOCALIDAD"
            ],

            "NOMBRE_LOCALIDAD": localidad,

            "GRUPO_ESPACIAL": grupo,

            "VIA_1": via_1,

            "VIA_2": via_2,

            "PUNTOS_ORIGINALES": fila[
                "PUNTOS_ORIGINALES"
            ],

            "LONGITUD_ORIGINAL": longitud_original,

            "LATITUD_ORIGINAL": latitud_original,

            "LONGITUD_CALCULADA": mejor[
                "LONGITUD_CRUCE"
            ],

            "LATITUD_CALCULADA": mejor[
                "LATITUD_CRUCE"
            ],

            "DISTANCIA_METROS": distancia,

            "CANTIDAD_CRUCES": len(cruces_df),

            "OBJECTID_VIA_1": mejor[
                "OBJECTID_VIA_1"
            ],

            "OBJECTID_VIA_2": mejor[
                "OBJECTID_VIA_2"
            ],

            "ESTADO": estado,

            "VALIDADO": "NO"

        })

    else:

        # No se encontraron cruces
        resultados_resumen.append({

            "DIRECCION": direccion,

            "CODIGO_LOCALIDAD": fila[
                "CODIGO_LOCALIDAD"
            ],

            "NOMBRE_LOCALIDAD": localidad,

            "GRUPO_ESPACIAL": grupo,

            "VIA_1": via_1,

            "VIA_2": via_2,

            "PUNTOS_ORIGINALES": fila[
                "PUNTOS_ORIGINALES"
            ],

            "LONGITUD_ORIGINAL": longitud_original,

            "LATITUD_ORIGINAL": latitud_original,

            "LONGITUD_CALCULADA": None,

            "LATITUD_CALCULADA": None,

            "DISTANCIA_METROS": None,

            "CANTIDAD_CRUCES": 0,

            "OBJECTID_VIA_1": None,

            "OBJECTID_VIA_2": None,

            "ESTADO": "SIN CRUCE ENCONTRADO",

            "VALIDADO": "NO"

        })

# =========================================================
# 8. EXPORTAR RESULTADOS
# =========================================================

resumen = pd.DataFrame(
    resultados_resumen
)

detalle = pd.DataFrame(
    resultados_detalle
)

resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8-sig"
)

detalle.to_csv(
    SALIDA_DETALLE,
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 9. INFORME FINAL
# =========================================================

print("\n" + "=" * 75)
print("RESULTADO DE LA GEORREFERENCIACIÓN AUTOMATIZADA")
print("=" * 75)

print("\nGrupos procesados:", len(resumen))

print(
    "Grupos con cruce encontrado:",
    int((resumen["CANTIDAD_CRUCES"] > 0).sum())
)

print(
    "Grupos sin cruce:",
    int((resumen["CANTIDAD_CRUCES"] == 0).sum())
)

print("\nClasificación de resultados:")

print(
    resumen["ESTADO"]
    .value_counts()
    .to_string()
)

print("\nPrimeros 20 resultados:")

columnas_mostrar = [
    "DIRECCION",
    "NOMBRE_LOCALIDAD",
    "GRUPO_ESPACIAL",
    "CANTIDAD_CRUCES",
    "DISTANCIA_METROS",
    "ESTADO"
]

print(
    resumen[columnas_mostrar]
    .head(20)
    .to_string(index=False)
)

print("\nArchivos generados:")

print(SALIDA_RESUMEN)
print(SALIDA_DETALLE)

print("\nIMPORTANTE:")
print(
    "Todos los resultados conservan VALIDADO = NO."
)
print(
    "La coincidencia geométrica requiere revisión "
    "antes de aprobar una coordenada."
)

print("\nProceso terminado.")
