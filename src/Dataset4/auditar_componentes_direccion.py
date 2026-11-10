
from pathlib import Path
import re
import pandas as pd

# ==========================================================
# CONFIGURACION
# ==========================================================

BASE = Path(__file__).resolve().parents[2]

ENTRADA = (
    BASE / "data" / "processed" / "siniestros_integrados.csv"
)

SALIDA_DETALLE = (
    BASE / "models" / "componentes_direcciones.csv"
)

SALIDA_RESUMEN = (
    BASE / "models" / "resumen_componentes_direcciones.csv"
)

# ==========================================================
# NORMALIZACION
# ==========================================================

def normalizar(texto):
    if pd.isna(texto):
        return ""

    texto = str(texto).upper().strip()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s*-\s*", "-", texto)

    # Retirar código final de localidad, por ejemplo 02
    texto = re.sub(r"\s+0*2$", "", texto)

    return texto.strip()


def limpiar_prefijo(texto):
    texto = texto.strip()

    texto = re.sub(
        r"^(AV\s+)?AVENIDA\s+",
        "",
        texto
    )

    texto = re.sub(
        r"^AV\s+",
        "",
        texto
    )

    return texto.strip()


# ==========================================================
# IDENTIFICACION DE VIAS
# ==========================================================

PATRON_NUMERICO = re.compile(
    r"^(KR|CR|CL|AC|AK|AV|TV|DG|DI)"
    r"\s*(\d+[A-Z]?)"
    r"\s*(S)?$"
)

ALIASES_NOMBRES = {
    "BOYACA": "AVENIDA BOYACA",
    "DEL SUR": "AVENIDA DEL SUR",
    "DE LAS AMERICAS": "AVENIDA DE LAS AMERICAS",
    "LAS AMERICAS": "AVENIDA DE LAS AMERICAS",
    "CIUDAD DE CALI": "AVENIDA CIUDAD DE CALI",
    "CIUDAD DE VILLAVICENCIO": "AVENIDA CIUDAD DE VILLAVICENCIO",
    "PRIMERO DE MAYO": "AVENIDA PRIMERO DE MAYO",
    "SUBA": "AVENIDA SUBA",
    "CARACAS": "AVENIDA CARACAS",
    "LUIS CARLOS GALAN": "AVENIDA LUIS CARLOS GALAN",
    "JORGE GAITAN CORTES": "AVENIDA JORGE GAITAN CORTES",
    "CIRCUNVALAR": "AVENIDA CIRCUNVALAR",
    "EL DORADO": "AVENIDA EL DORADO",
    "BOYACÁ": "AVENIDA BOYACA"
}


def analizar_componente(componente):

    componente = limpiar_prefijo(componente)

    if not componente:
        return "VACIO", "", ""

    # Nomenclatura numérica
    coincidencia = PATRON_NUMERICO.fullmatch(
        componente
    )

    if coincidencia:

        tipo, numero, sur = coincidencia.groups()

        nombre = tipo + numero

        if sur:
            nombre += " S"

        return (
            "VIA_NUMERICA",
            nombre,
            nombre
        )

    # Vías con nombres especiales
    for alias, nombre_oficial in ALIASES_NOMBRES.items():

        if componente == alias:
            return (
                "VIA_NOMBRE",
                nombre_oficial,
                nombre_oficial
            )

    # Otros nombres de vía que requieren comparación
    if re.search(r"[A-Z]", componente):

        return (
            "VIA_NOMBRE_PENDIENTE_CATALOGO",
            componente,
            componente
        )

    return (
        "NO_RECONOCIDA",
        componente,
        ""
    )


# ==========================================================
# PROCESAMIENTO
# ==========================================================

print("\n==============================================")
print(" AUDITORIA DE COMPONENTES DE DIRECCIONES")
print("==============================================")

if not ENTRADA.exists():
    raise FileNotFoundError(ENTRADA)

df = pd.read_csv(
    ENTRADA,
    encoding="utf-8-sig",
    low_memory=False
)

print(f"\nRegistros originales: {len(df)}")

if "DIRECCION" not in df.columns:
    raise ValueError("No existe la columna DIRECCION")

if "CODIGO_LOCALIDAD" not in df.columns:
    raise ValueError("No existe CODIGO_LOCALIDAD")

df["DIRECCION_NORMALIZADA"] = (
    df["DIRECCION"].apply(normalizar)
)

# Separar en los dos componentes
partes = df["DIRECCION_NORMALIZADA"].str.split(
    "-",
    n=1,
    expand=True
)

df["COMPONENTE_1"] = partes[0].fillna("").str.strip()
df["COMPONENTE_2"] = partes[1].fillna("").str.strip()

# Analizar componente 1
resultado_1 = df["COMPONENTE_1"].apply(
    analizar_componente
)

df["TIPO_VIA_1"] = resultado_1.apply(
    lambda x: x[0]
)

df["VIA_1_NORMALIZADA"] = resultado_1.apply(
    lambda x: x[1]
)

df["CLAVE_VIA_1"] = resultado_1.apply(
    lambda x: x[2]
)

# Analizar componente 2
resultado_2 = df["COMPONENTE_2"].apply(
    analizar_componente
)

df["TIPO_VIA_2"] = resultado_2.apply(
    lambda x: x[0]
)

df["VIA_2_NORMALIZADA"] = resultado_2.apply(
    lambda x: x[1]
)

df["CLAVE_VIA_2"] = resultado_2.apply(
    lambda x: x[2]
)

# ==========================================================
# CLASIFICACION GENERAL
# ==========================================================

def clasificar_fila(fila):

    tipo1 = fila["TIPO_VIA_1"]
    tipo2 = fila["TIPO_VIA_2"]

    if tipo1 == "VACIO" or tipo2 == "VACIO":
        return "FALTA_COMPONENTE"

    if (
        tipo1 in ["VIA_NUMERICA", "VIA_NOMBRE"]
        and tipo2 in ["VIA_NUMERICA", "VIA_NOMBRE"]
    ):
        return "DOS_VIAS_RECONOCIDAS"

    if (
        tipo1 == "VIA_NUMERICA"
        and tipo2 == "VIA_NOMBRE_PENDIENTE_CATALOGO"
    ) or (
        tipo2 == "VIA_NUMERICA"
        and tipo1 == "VIA_NOMBRE_PENDIENTE_CATALOGO"
    ):
        return "UNA_NUMERICA_UN_NOMBRE_PENDIENTE"

    if (
        tipo1 == "VIA_NUMERICA"
        or tipo2 == "VIA_NUMERICA"
    ):
        return "UNA_VIA_NUMERICA"

    if (
        tipo1 == "VIA_NOMBRE"
        or tipo2 == "VIA_NOMBRE"
    ):
        return "VIAS_CON_NOMBRE"

    return "REQUIERE_REVISION"


df["CLASIFICACION_COMPONENTES"] = df.apply(
    clasificar_fila,
    axis=1
)

# ==========================================================
# DETALLE POR DIRECCION Y LOCALIDAD
# ==========================================================

columnas_detalle = [
    "CODIGO_LOCALIDAD",
    "DIRECCION",
    "DIRECCION_NORMALIZADA",
    "COMPONENTE_1",
    "TIPO_VIA_1",
    "VIA_1_NORMALIZADA",
    "COMPONENTE_2",
    "TIPO_VIA_2",
    "VIA_2_NORMALIZADA",
    "CLASIFICACION_COMPONENTES"
]

detalle = (
    df.groupby(
        columnas_detalle,
        dropna=False
    )
    .size()
    .reset_index(name="TOTAL_SINIESTROS")
    .sort_values(
        "TOTAL_SINIESTROS",
        ascending=False
    )
)

# ==========================================================
# RESUMEN
# ==========================================================

resumen = (
    detalle.groupby(
        "CLASIFICACION_COMPONENTES"
    )
    .agg(
        DIRECCIONES=("DIRECCION_NORMALIZADA", "nunique"),
        COMBINACIONES_LOCALIDAD_DIRECCION=(
            "DIRECCION_NORMALIZADA",
            "size"
        ),
        TOTAL_SINIESTROS=("TOTAL_SINIESTROS", "sum")
    )
    .reset_index()
    .sort_values(
        "TOTAL_SINIESTROS",
        ascending=False
    )
)

# ==========================================================
# EXPORTAR
# ==========================================================

detalle.to_csv(
    SALIDA_DETALLE,
    index=False,
    encoding="utf-8-sig"
)

resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# INFORME
# ==========================================================

print("\n==============================================")
print(" RESULTADOS")
print("==============================================")

print("\nRegistros originales:", len(df))

print(
    "Direcciones distintas:",
    df["DIRECCION_NORMALIZADA"].nunique()
)

print(
    "Combinaciones dirección-localidad:",
    detalle.shape[0]
)

print("\nCLASIFICACION DE COMPONENTES:")

print(
    resumen.to_string(index=False)
)

print("\nEJEMPLOS DE DIRECCIONES CON DOS VIAS RECONOCIDAS:")

print(
    detalle[
        detalle["CLASIFICACION_COMPONENTES"]
        == "DOS_VIAS_RECONOCIDAS"
    ][
        [
            "DIRECCION",
            "VIA_1_NORMALIZADA",
            "VIA_2_NORMALIZADA",
            "TOTAL_SINIESTROS"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nDIRECCIONES QUE REQUIEREN REVISION:")

print(
    detalle[
        detalle["CLASIFICACION_COMPONENTES"]
        == "REQUIERE_REVISION"
    ][
        [
            "DIRECCION",
            "COMPONENTE_1",
            "COMPONENTE_2",
            "TOTAL_SINIESTROS"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nARCHIVOS GENERADOS:")
print(SALIDA_DETALLE)
print(SALIDA_RESUMEN)

print("\nIMPORTANTE:")
print(
    "La identificacion de nombres de vias es preliminar. "
    "Todavia no representa una coincidencia validada "
    "con la geometria de la malla vial."
)

print("\nProceso terminado.")
