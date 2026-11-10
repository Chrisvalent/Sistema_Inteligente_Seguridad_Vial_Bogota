
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

    texto = re.sub(r"[^A-Z0-9 ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def extraer_via(texto):
    texto = normalizar(texto)

    equivalencias = {
        "AVENIDA BOYACA": "AVENIDA BOYACA",
        "AVENIDA DE LAS AMERICAS": "AVENIDA DE LAS AMERICAS",
        "AVENIDA CIUDAD DE CALI": "AVENIDA CIUDAD DE CALI",
        "AVENIDA DEL SUR": "AVENIDA DEL SUR",
        "AVENIDA CARACAS": "AVENIDA CARACAS",
        "AVENIDA SUBA": "AVENIDA SUBA",
        "AVENIDA PRIMERO DE MAYO": "AVENIDA PRIMERO DE MAYO",
        "AVENIDA CIRCUNVALAR": "AVENIDA CIRCUNVALAR"
    }

    for nombre in equivalencias:
        if nombre in texto:
            return equivalencias[nombre]

    patron = (
        r"\b(CL|KR|CRA|CR|AK|AC|TV|DG)\s*"
        r"(\d+[A-Z]*(?:BIS)?)"
    )

    coincidencia = re.search(patron, texto)

    if coincidencia:
        tipo = coincidencia.group(1)
        numero = coincidencia.group(2)

        if tipo in ["CRA", "CR"]:
            tipo = "KR"

        return f"{tipo} {numero}"

    return None


print("=" * 75)
print("AUDITORIA DE AVENIDAS Y NOMENCLATURAS")
print("=" * 75)

malla = gpd.read_file(
    RUTA_MALLA,
    engine="pyogrio"
)

muestra = pd.read_csv(
    RUTA_MUESTRA,
    dtype={"DIRECCION": str}
)

# Catálogo de nombres de vías
catalogo = set()

for campo in ["MVINOMBRE", "MVINALTERN"]:

    for valor in malla[campo].dropna().unique():

        nombre = normalizar(valor)

        if nombre:
            catalogo.add(nombre)

# También agregar nomenclaturas
for campo in ["MVINANTIGU", "MVIETIQUET"]:

    for valor in malla[campo].dropna().unique():

        nombre = extraer_via(valor)

        if nombre:
            catalogo.add(nombre)

resultados = []

for _, fila in muestra.iterrows():

    direccion = str(fila["DIRECCION"])
    partes = direccion.split("-", 1)

    vias = [extraer_via(parte) for parte in partes]

    coincidencias = []

    for via in vias:

        if via and via in catalogo:
            coincidencias.append(True)
        else:
            coincidencias.append(False)

    cantidad = sum(coincidencias)

    if cantidad == 2:
        estado = "DOS VIAS IDENTIFICADAS"
    elif cantidad == 1:
        estado = "UNA VIA IDENTIFICADA"
    else:
        estado = "NINGUNA VIA IDENTIFICADA"

    resultados.append({
        "DIRECCION": direccion,
        "VIA_1": vias[0],
        "VIA_2": vias[1] if len(vias) > 1 else None,
        "COINCIDENCIAS": cantidad,
        "ESTADO": estado
    })

resultado = pd.DataFrame(resultados)

print("\nRESUMEN")
print(resultado["ESTADO"].value_counts().to_string())

print("\nPRIMERAS 30 DIRECCIONES")
print(resultado.head(30).to_string(index=False))

resultado.to_csv(
    "models/auditoria_avenidas_principales.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:")
print("models/auditoria_avenidas_principales.csv")

print("\nAuditoria terminada.")
