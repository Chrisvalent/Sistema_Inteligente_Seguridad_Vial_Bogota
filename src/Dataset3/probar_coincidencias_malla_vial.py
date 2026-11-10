
import geopandas as gpd
import pandas as pd
import re
import unicodedata

RUTA_MALLA = "data/raw/Malla_Vial_Integral_Bogota_D_C.geojson"
RUTA_MUESTRA = "data/processed/muestra_georreferenciacion.csv"

print("=" * 75)
print("PRUEBA DE COINCIDENCIAS CON LA MALLA VIAL")
print("=" * 75)


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


# Cargar datos
print("\nCargando muestra...")
muestra = pd.read_csv(RUTA_MUESTRA, dtype={"DIRECCION": str})

print("Direcciones:", len(muestra))

print("\nCargando malla vial...")
malla = gpd.read_file(
    RUTA_MALLA,
    engine="pyogrio"
)

print("Segmentos:", len(malla))

# Campos que pueden contener nomenclatura
campos = [
    "MVINANTIGU",
    "MVIETIQUET",
    "MVINOMBRE",
    "MVINALTERN"
]

# Construir catálogo normalizado
catalogo = []

for _, fila in malla.iterrows():

    for campo in campos:

        if campo not in malla.columns:
            continue

        valor = normalizar(fila[campo])

        if valor:
            catalogo.append({
                "nomenclatura": valor,
                "campo": campo,
                "OBJECTID": fila["OBJECTID"]
            })

catalogo = pd.DataFrame(catalogo).drop_duplicates()

print("Referencias de nomenclatura:", len(catalogo))

# Analizar direcciones de la muestra
resultados = []

for _, fila in muestra.iterrows():

    direccion = str(fila["DIRECCION"])
    partes = direccion.split("-", 1)

    coincidencias = []

    for parte in partes:

        parte_norm = normalizar(parte)

        if not parte_norm:
            continue

        # Buscar referencias exactas o contenidas
        encontrados = catalogo[
            catalogo["nomenclatura"].apply(
                lambda x: (
                    x == parte_norm
                    or x in parte_norm
                    or parte_norm in x
                )
            )
        ]

        for _, encontrado in encontrados.iterrows():
            coincidencias.append({
                "parte": parte_norm,
                "nomenclatura": encontrado["nomenclatura"],
                "campo": encontrado["campo"],
                "OBJECTID": encontrado["OBJECTID"]
            })

    resultados.append({
        "DIRECCION": direccion,
        "CODIGO_LOCALIDAD": fila["CODIGO_LOCALIDAD"],
        "ACCIDENTES": fila["cantidad_accidentes"],
        "COINCIDENCIAS": len(coincidencias),
        "DETALLE": str(coincidencias[:5])
    })

resultado = pd.DataFrame(resultados)

# Clasificación preliminar
resultado["ESTADO"] = resultado["COINCIDENCIAS"].apply(
    lambda x: (
        "CON COINCIDENCIAS"
        if x > 0
        else "SIN COINCIDENCIAS"
    )
)

print("\n" + "=" * 75)
print("RESULTADOS")
print("=" * 75)

print(
    resultado[
        [
            "DIRECCION",
            "CODIGO_LOCALIDAD",
            "ACCIDENTES",
            "COINCIDENCIAS",
            "ESTADO"
        ]
    ].to_string(index=False)
)

print("\nResumen:")
print(resultado["ESTADO"].value_counts().to_string())

# Guardar resultados
resultado.to_csv(
    "models/coincidencias_malla_vial.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:")
print("models/coincidencias_malla_vial.csv")

print("\nPrueba terminada.")
print(
    "\nNOTA: una coincidencia textual no demuestra "
    "que se haya localizado correctamente una intersección."
)
