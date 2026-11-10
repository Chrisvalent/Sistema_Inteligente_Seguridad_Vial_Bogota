
import pandas as pd
import re
import os

RUTA = "data/processed/siniestros_integrados.csv"
SALIDA = "models/auditoria_estructura_direcciones.csv"

print("=" * 75)
print("AUDITORIA DE ESTRUCTURA DE DIRECCIONES")
print("=" * 75)

df = pd.read_csv(RUTA, low_memory=False)

direcciones = (
    df["DIRECCION"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.strip()
)

# Tipos de vías y abreviaturas frecuentes
tipos_via = (
    r"(?:CL|KR|CRA|CR|AK|AC|AV|DG|"
    r"TV|TR|AUTOPISTA|AVENIDA|CALLE|CARRERA|DIAGONAL|TRANSVERSAL)"
)

# Detectar dos referencias viales separadas por guion
patron_cruce = re.compile(
    rf"^\s*{tipos_via}\b.*-\s*{tipos_via}\b",
    re.IGNORECASE
)

# Detectar presencia de una referencia vial
patron_vial = re.compile(
    rf"\b{tipos_via}\b",
    re.IGNORECASE
)

cruces = direcciones.apply(
    lambda x: bool(patron_cruce.search(x)) if x else False
)

referencias_viales = direcciones.apply(
    lambda x: bool(patron_vial.search(x)) if x else False
)

# Clasificación preliminar
def clasificar(direccion):
    if not direccion:
        return "Vacia"

    if patron_cruce.search(direccion):
        return "Posible cruce vial"

    if patron_vial.search(direccion):
        return "Referencia vial sin cruce reconocido"

    return "Estructura no reconocida"


clasificaciones = direcciones.apply(clasificar)

print("\n1. RESUMEN")
print("Registros:", len(direcciones))
print("Direcciones únicas:", direcciones.nunique())
print("Posibles cruces viales:", int(cruces.sum()))
print(
    "Porcentaje de posibles cruces:",
    round(cruces.mean() * 100, 2), "%"
)
print(
    "Con referencia vial:",
    int(referencias_viales.sum())
)

print("\n2. CLASIFICACION DE DIRECCIONES")
print(clasificaciones.value_counts().to_string())

print("\n3. EJEMPLOS DE POSIBLES CRUCES")
for direccion in direcciones[cruces].drop_duplicates().head(30):
    print("-", direccion)

print("\n4. EJEMPLOS NO RECONOCIDOS")
no_reconocidas = direcciones[
    clasificaciones == "Estructura no reconocida"
]

for direccion in no_reconocidas.drop_duplicates().head(20):
    print("-", direccion)

# Guardar clasificación por registro
resultado = pd.DataFrame({
    "DIRECCION": direcciones,
    "CLASIFICACION": clasificaciones
})

os.makedirs("models", exist_ok=True)

resultado.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:", SALIDA)
print("\nAuditoria terminada.")
