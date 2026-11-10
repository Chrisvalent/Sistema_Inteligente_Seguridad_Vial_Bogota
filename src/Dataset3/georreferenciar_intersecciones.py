
import re
import unicodedata
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union


# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================

BASE = Path(__file__).resolve().parents[2]

ARCHIVO_MUESTRA = (
    BASE / "data" / "processed" / "muestra_georreferenciacion.csv"
)

ARCHIVO_MALLA = (
    BASE / "data" / "raw" / "Malla_Vial_Integral_Bogota_D_C.geojson"
)

SALIDA_CANDIDATOS = (
    BASE / "models" / "candidatos_intersecciones.csv"
)

SALIDA_RESUMEN = (
    BASE / "models" / "resumen_georreferenciacion.csv"
)

SALIDA_VIAS = (
    BASE / "models" / "auditoria_coincidencias_malla.csv"
)


# ============================================================
# 2. NORMALIZACIÓN DE TEXTO
# ============================================================

def normalizar(valor):

    if pd.isna(valor):
        return ""

    texto = str(valor).upper().strip()

    texto = unicodedata.normalize("NFKD", texto)

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    texto = re.sub(r"[^A-Z0-9]", "", texto)

    return texto


# ============================================================
# 3. IDENTIFICAR LAS DOS VÍAS DE CADA DIRECCIÓN
# ============================================================

NOMBRES_AVENIDAS = [
    "AVENIDA DE LAS AMERICAS",
    "AVENIDA CIUDAD DE CALI",
    "AVENIDA PRIMERO DE MAYO",
    "AVENIDA BOYACA",
    "AVENIDA DEL SUR",
    "AVENIDA CARACAS",
    "AVENIDA CIRCUNVALAR",
    "AVENIDA SUBA",
]


def extraer_via(texto):

    texto = str(texto).upper().strip()

    texto_normalizado = normalizar(texto)

    # Primero se buscan las avenidas por nombre.
    for avenida in NOMBRES_AVENIDAS:

        if normalizar(avenida) in texto_normalizado:

            return avenida

    # Después se buscan las nomenclaturas convencionales.
    # Ejemplos:
    # KR 80
    # CL 13
    # CL 59 S
    # AK 45
    # AC 26

    coincidencia = re.search(
        r"\b(KR|CL|AK|AC|TV|DG)\s*(\d+)\s*(S)?\b",
        texto
    )

    if coincidencia:

        tipo = coincidencia.group(1)

        numero = coincidencia.group(2)

        sur = coincidencia.group(3) or ""

        return f"{tipo} {numero} {sur}".strip()

    return ""


def vias_de_direccion(direccion):

    partes = str(direccion).upper().split("-", 1)

    if len(partes) != 2:

        return "", ""

    via1 = extraer_via(partes[0])

    via2 = extraer_via(partes[1])

    return via1, via2


# ============================================================
# 4. PREPARAR LOS NOMBRES DE LA MALLA VIAL
# ============================================================

def preparar_catalogo_vial(malla):

    campos = [
        "MVINANTIGU",
        "MVIETIQUET",
        "MVINOMBRE",
        "MVINALTERN",
    ]

    for campo in campos:

        if campo not in malla.columns:

            malla[campo] = ""

    catalogo = []

    for indice, fila in malla.iterrows():

        for campo in campos:

            valor_original = fila.get(campo, "")

            valor_normalizado = normalizar(valor_original)

            if not valor_normalizado:

                continue

            catalogo.append({
                "INDICE": indice,
                "CAMPO": campo,
                "VALOR_ORIGINAL": str(valor_original),
                "VALOR_NORMALIZADO": valor_normalizado,
            })

    return pd.DataFrame(catalogo)


# ============================================================
# 5. BUSCAR UNA VÍA EN LA MALLA
# ============================================================

def buscar_segmentos(via, malla, catalogo):

    objetivo = normalizar(via)

    if not objetivo:

        return malla.iloc[0:0].copy(), "VIA VACIA"

    coincidencias = catalogo[
        catalogo["VALOR_NORMALIZADO"] == objetivo
    ]

    if coincidencias.empty:

        return malla.iloc[0:0].copy(), "NO ENCONTRADA"

    indices = coincidencias["INDICE"].drop_duplicates().tolist()

    segmentos = malla.loc[indices].copy()

    return segmentos, "ENCONTRADA"


# ============================================================
# 6. EXTRAER PUNTOS DE UNA INTERSECCIÓN GEOMÉTRICA
# ============================================================

def obtener_puntos(geometria):

    if geometria is None or geometria.is_empty:

        return []

    tipo = geometria.geom_type

    if tipo == "Point":

        return [geometria]

    if tipo == "MultiPoint":

        return list(geometria.geoms)

    if tipo == "GeometryCollection":

        puntos = []

        for elemento in geometria.geoms:

            puntos.extend(obtener_puntos(elemento))

        return puntos

    # Una intersección lineal no se convierte
    # artificialmente en un punto.
    return []


# ============================================================
# 7. CALCULAR INTERSECCIONES ENTRE DOS VÍAS
# ============================================================

def calcular_intersecciones(segmentos1, segmentos2):

    geometria1 = [
        geometria
        for geometria in segmentos1.geometry
        if geometria is not None and not geometria.is_empty
    ]

    geometria2 = [
        geometria
        for geometria in segmentos2.geometry
        if geometria is not None and not geometria.is_empty
    ]

    if not geometria1 or not geometria2:

        return []

    union1 = unary_union(geometria1)

    union2 = unary_union(geometria2)

    interseccion = union1.intersection(union2)

    return obtener_puntos(interseccion)


# ============================================================
# 8. PROCESO PRINCIPAL
# ============================================================

def main():

    print("=" * 75)

    print("GEORREFERENCIACIÓN DE INTERSECCIONES VIALES")

    print("=" * 75)

    # --------------------------------------------------------
    # Verificar archivos
    # --------------------------------------------------------

    if not ARCHIVO_MUESTRA.exists():

        raise FileNotFoundError(
            f"No existe el archivo de muestra: {ARCHIVO_MUESTRA}"
        )

    if not ARCHIVO_MALLA.exists():

        raise FileNotFoundError(
            f"No existe el GeoJSON de la malla vial: {ARCHIVO_MALLA}"
        )

    # --------------------------------------------------------
    # Leer muestra
    # --------------------------------------------------------

    muestra = pd.read_csv(
        ARCHIVO_MUESTRA,
        encoding="utf-8-sig"
    )

    columnas_necesarias = [
        "DIRECCION",
        "CODIGO_LOCALIDAD",
    ]

    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in muestra.columns
    ]

    if faltantes:

        raise ValueError(
            f"Faltan columnas en la muestra: {faltantes}"
        )

    print(
        f"\nDirecciones cargadas: {len(muestra):,}"
    )

    # --------------------------------------------------------
    # Leer GeoJSON
    # --------------------------------------------------------

    print("\nCargando malla vial...")

    malla = gpd.read_file(ARCHIVO_MALLA)

    print(
        f"Segmentos cargados: {len(malla):,}"
    )

    print(
        f"CRS original: {malla.crs}"
    )

    if malla.crs is None:

        raise ValueError(
            "La malla vial no tiene sistema de coordenadas definido."
        )

    if malla.crs.to_epsg() != 4326:

        malla = malla.to_crs(epsg=4326)

    print(
        f"CRS utilizado: {malla.crs}"
    )

    # --------------------------------------------------------
    # Preparar catálogo de vías
    # --------------------------------------------------------

    print("\nConstruyendo catálogo de nomenclaturas...")

    catalogo = preparar_catalogo_vial(malla)

    print(
        f"Registros de nomenclatura: {len(catalogo):,}"
    )

    # --------------------------------------------------------
    # Procesar direcciones
    # --------------------------------------------------------

    resultados = []

    candidatos = []

    cache = {}

    for numero, fila in muestra.iterrows():

        direccion = str(fila["DIRECCION"])

        localidad = fila["CODIGO_LOCALIDAD"]

        via1, via2 = vias_de_direccion(direccion)

        print(
            f"\n[{numero + 1}/{len(muestra)}] {direccion}"
        )

        # --------------------------------------------
        # Verificar extracción
        # --------------------------------------------

        if not via1 or not via2:

            resultados.append({

                "DIRECCION": direccion,

                "CODIGO_LOCALIDAD": localidad,

                "VIA_1": via1,

                "VIA_2": via2,

                "ESTADO_VIA_1": "NO EXTRAIDA" if not via1 else "",

                "ESTADO_VIA_2": "NO EXTRAIDA" if not via2 else "",

                "SEGMENTOS_VIA_1": 0,

                "SEGMENTOS_VIA_2": 0,

                "CANDIDATOS": 0,

                "ESTADO": "ERROR EN EXTRACCION",

            })

            print("  No se pudieron extraer las dos vías.")

            continue

        # --------------------------------------------
        # Usar caché para parejas repetidas
        # --------------------------------------------

        clave = tuple(
            sorted([
                normalizar(via1),
                normalizar(via2)
            ])
        )

        if clave not in cache:

            segmentos1, estado1 = buscar_segmentos(
                via1,
                malla,
                catalogo
            )

            segmentos2, estado2 = buscar_segmentos(
                via2,
                malla,
                catalogo
            )

            puntos = []

            if estado1 == "ENCONTRADA" and estado2 == "ENCONTRADA":

                puntos = calcular_intersecciones(
                    segmentos1,
                    segmentos2
                )

            cache[clave] = {

                "segmentos1": segmentos1,

                "segmentos2": segmentos2,

                "estado1": estado1,

                "estado2": estado2,

                "puntos": puntos,

            }

        resultado = cache[clave]

        segmentos1 = resultado["segmentos1"]

        segmentos2 = resultado["segmentos2"]

        puntos = resultado["puntos"]

        estado1 = resultado["estado1"]

        estado2 = resultado["estado2"]

        # --------------------------------------------
        # Determinar estado
        # --------------------------------------------

        if estado1 != "ENCONTRADA" or estado2 != "ENCONTRADA":

            estado = "VIA NO ENCONTRADA EN MALLA"

        elif len(puntos) == 0:

            estado = "SIN INTERSECCION PUNTUAL"

        elif len(puntos) == 1:

            estado = "UN CANDIDATO"

        else:

            estado = "VARIOS CANDIDATOS - VALIDAR"

        # --------------------------------------------
        # Guardar resumen
        # --------------------------------------------

        resultados.append({

            "DIRECCION": direccion,

            "CODIGO_LOCALIDAD": localidad,

            "VIA_1": via1,

            "VIA_2": via2,

            "ESTADO_VIA_1": estado1,

            "ESTADO_VIA_2": estado2,

            "SEGMENTOS_VIA_1": len(segmentos1),

            "SEGMENTOS_VIA_2": len(segmentos2),

            "CANDIDATOS": len(puntos),

            "ESTADO": estado,

        })

        print(
            f"  Vía 1: {via1} -> {estado1} "
            f"({len(segmentos1)} segmentos)"
        )

        print(
            f"  Vía 2: {via2} -> {estado2} "
            f"({len(segmentos2)} segmentos)"
        )

        print(
            f"  Candidatos geométricos: {len(puntos)}"
        )

        # --------------------------------------------
        # Guardar coordenadas candidatas
        # --------------------------------------------

        for indice, punto in enumerate(puntos, start=1):

            candidatos.append({

                "DIRECCION": direccion,

                "CODIGO_LOCALIDAD": localidad,

                "VIA_1": via1,

                "VIA_2": via2,

                "CANDIDATO": indice,

                "LONGITUD": punto.x,

                "LATITUD": punto.y,

                "CRS": "EPSG:4326",

                "VALIDADO": "NO",

            })

    # ========================================================
    # 9. EXPORTAR RESULTADOS
    # ========================================================

    df_resultados = pd.DataFrame(resultados)

    df_candidatos = pd.DataFrame(candidatos)

    df_resultados.to_csv(
        SALIDA_RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    df_candidatos.to_csv(
        SALIDA_CANDIDATOS,
        index=False,
        encoding="utf-8-sig"
    )

    # Auditoría de las vías reconocidas en la muestra.
    auditoria_vias = df_resultados[
        [
            "DIRECCION",
            "VIA_1",
            "VIA_2",
            "ESTADO_VIA_1",
            "ESTADO_VIA_2",
            "SEGMENTOS_VIA_1",
            "SEGMENTOS_VIA_2",
            "ESTADO",
        ]
    ].copy()

    auditoria_vias.to_csv(
        SALIDA_VIAS,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 10. INFORME FINAL
    # ========================================================

    print("\n" + "=" * 75)

    print("RESUMEN FINAL")

    print("=" * 75)

    print("\nEstados de georreferenciación:")

    print(
        df_resultados["ESTADO"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTotal de direcciones:", len(df_resultados))

    print("Total de candidatos:", len(df_candidatos))

    print(
        "\nDirecciones con ambas vías reconocidas:",
        (
            (df_resultados["ESTADO_VIA_1"] == "ENCONTRADA")
            &
            (df_resultados["ESTADO_VIA_2"] == "ENCONTRADA")
        ).sum()
    )

    print("\nPrimeras 20 direcciones:")

    print(
        df_resultados.head(20).to_string(index=False)
    )

    print("\nArchivos generados:")

    print(SALIDA_CANDIDATOS)

    print(SALIDA_RESUMEN)

    print(SALIDA_VIAS)

    print("\nProceso terminado.")


if __name__ == "__main__":

    main()
