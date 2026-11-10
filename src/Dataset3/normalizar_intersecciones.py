
import pandas as pd
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

ARCHIVO = (
    BASE / "models" /
    "candidatos_agrupados_espacialmente.csv"
)

SALIDA_DETALLE = (
    BASE / "models" /
    "intersecciones_candidatos_normalizados.csv"
)

SALIDA_RESUMEN = (
    BASE / "models" /
    "resumen_intersecciones_normalizadas.csv"
)

# --------------------------------------------------
# 1. CARGAR INFORMACIÓN
# --------------------------------------------------

df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

columnas_requeridas = [
    "DIRECCION",
    "CODIGO_LOCALIDAD",
    "NOMBRE_LOCALIDAD",
    "VIA_1",
    "VIA_2",
    "GRUPO_ESPACIAL",
    "PUNTOS_ORIGINALES",
    "LONGITUD_CENTRO",
    "LATITUD_CENTRO",
    "DISTANCIA_AGRUPACION_METROS",
    "VALIDADO"
]

faltantes = [
    c for c in columnas_requeridas
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas requeridas: {faltantes}"
    )

# --------------------------------------------------
# 2. NORMALIZAR DIRECCIONES
# --------------------------------------------------

def normalizar_texto(valor):
    texto = str(valor).upper().strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto

def normalizar_numero_final(texto):
    # Normaliza únicamente el último número
    # Ejemplo: "... 02" pasa a "... 2"
    return re.sub(
        r"\s+0+(\d+)$",
        r" \1",
        texto
    )

df["DIRECCION_NORMALIZADA"] = (
    df["DIRECCION"]
    .apply(normalizar_texto)
    .apply(normalizar_numero_final)
)

df["VIA_1_NORMALIZADA"] = (
    df["VIA_1"].apply(normalizar_texto)
)

df["VIA_2_NORMALIZADA"] = (
    df["VIA_2"].apply(normalizar_texto)
)

# --------------------------------------------------
# 3. CONSERVAR EL DETALLE DE CADA GRUPO
# --------------------------------------------------

# No promediamos ni eliminamos coordenadas.
# Cada grupo espacial permanece como candidato independiente.

df.to_csv(
    SALIDA_DETALLE,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 4. CREAR RESUMEN POR INTERSECCIÓN
# --------------------------------------------------

resumen = (
    df.groupby(
        [
            "DIRECCION_NORMALIZADA",
            "CODIGO_LOCALIDAD",
            "NOMBRE_LOCALIDAD"
        ],
        as_index=False
    )
    .agg(
        VARIANTES_DIRECCION=("DIRECCION", "nunique"),
        CANTIDAD_GRUPOS=("GRUPO_ESPACIAL", "count"),
        TOTAL_PUNTOS_ORIGINALES=("PUNTOS_ORIGINALES", "sum"),
        GRUPOS_VALIDOS=(
            "VALIDADO",
            lambda x: (x.astype(str).str.upper() == "SI").sum()
        )
    )
)

resumen["ESTADO_REVISION"] = resumen[
    "GRUPOS_VALIDOS"
].apply(
    lambda x: "CON VALIDACION" if x > 0 else "PENDIENTE"
)

resumen = resumen.sort_values(
    ["CANTIDAD_GRUPOS", "TOTAL_PUNTOS_ORIGINALES"],
    ascending=False
)

resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 5. INFORME
# --------------------------------------------------

print("=" * 70)
print("NORMALIZACIÓN DE INTERSECCIONES")
print("=" * 70)

print(f"\nGrupos originales: {len(df)}")

print(
    "Intersecciones normalizadas:",
    len(resumen)
)

print(
    "Variantes de dirección consolidadas:",
    int(resumen["VARIANTES_DIRECCION"].sum())
)

print(
    "Grupos espaciales conservados:",
    len(df)
)

print(
    "\nGrupos con validación:",
    int((df["VALIDADO"].astype(str).str.upper() == "SI").sum())
)

print("\nPrimeras 15 intersecciones:")
print(resumen.head(15).to_string(index=False))

print("\nArchivos generados:")
print(SALIDA_DETALLE)
print(SALIDA_RESUMEN)

print("\nProceso terminado.")
