
import geopandas as gpd
import pandas as pd
import re
import unicodedata

RUTA_MALLA = "data/raw/Malla_Vial_Integral_Bogota_D_C.geojson"
RUTA_MUESTRA = "data/processed/muestra_georreferenciacion.csv"


def normalizar(texto):
    if pd.isna(texto):
        return ""

    texto = str(texto).upper().strip()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def extraer_via(texto):
    texto = normalizar(texto)

    # Eliminar prefijos descriptivos de avenidas
    texto = re.sub(
        r"^AV\s+AVENIDA\s+",
        "",
        texto
    )

    # Reconocer nomenclaturas convencionales
    patron = (
        r"\b(CL|KR|CRA|CR|AK|AC|TV|DG|"
        r"AV|TRANSVERSAL|DIAGONAL)\s*"
        r"(\d+[A-Z]*(?:BIS)?)"
    )

    coincidencia = re.search(patron, texto)

    if not coincidencia:
        return None

    tipo = coincidencia.group(1)
    numero = coincidencia.group(2)

    equivalencias = {
        "CRA": "KR",
        "CR": "KR",
        "TRANSVERSAL": "TV",
        "DIAGONAL": "DG"
    }

    tipo = equivalencias.get(tipo, tipo)

    return f"{tipo} {numero}"


print("=" * 75)
print("VALIDACION EXACTA DE NOMENCLATURA")
print("=" * 75)

print("\nCargando malla vial...")
malla = gpd.read_file(
    RUTA_MALLA,
    engine="pyogrio"
)

print("Segmentos:", len(malla))

# Catálogo de nomenclaturas
catalogo = set()

for campo in ["MVINANTIGU", "MVIETIQUET"]:

    if campo not in malla.columns:
        continue

    for valor in malla[campo].dropna().unique():

        via = extraer_via(valor)

        if via:
            catalogo.add(via)

print("Nomenclaturas únicas reconocidas:", len(catalogo))

print("\nCargando muestra...")
muestra = pd.read_csv(
    RUTA_MUESTRA,
    dtype={"DIRECCION": str}
)

resultados = []

for _, fila in muestra.iterrows():

    direccion = str(fila["DIRECCION"])
    partes = direccion.split("-", 1)

    vias = [extraer_via(parte) for parte in partes]

    coincidencias = [
        via in catalogo if via else False
        for via in vias
    ]

    cantidad = sum(coincidencias)

    if cantidad == 2:
        estado = "DOS VIAS IDENTIFICADAS"
    elif cantidad == 1:
        estado = "UNA VIA IDENTIFICADA"
    else:
        estado = "NINGUNA VIA IDENTIFICADA"

    resultados.append({
        "DIRECCION": direccion,
        "CODIGO_LOCALIDAD": fila["CODIGO_LOCALIDAD"],
        "ACCIDENTES": fila["cantidad_accidentes"],
        "VIA_1": vias[0],
        "VIA_2": vias[1] if len(vias) > 1 else None,
        "COINCIDENCIAS_VIAS": cantidad,
        "ESTADO": estado
    })

resultado = pd.DataFrame(resultados)

print("\nRESULTADOS")
print("-" * 75)

print(
    resultado[
        [
            "DIRECCION",
            "VIA_1",
            "VIA_2",
            "COINCIDENCIAS_VIAS",
            "ESTADO"
        ]
    ].head(30).to_string(index=False)
)

print("\nRESUMEN")
print(resultado["ESTADO"].value_counts().to_string())

resultado.to_csv(
    "models/validacion_nomenclatura_exacta.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:")
print("models/validacion_nomenclatura_exacta.csv")

print("\nValidación terminada.")
