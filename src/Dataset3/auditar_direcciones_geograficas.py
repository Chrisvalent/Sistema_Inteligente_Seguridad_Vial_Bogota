
import pandas as pd
import re

RUTA = "data/processed/siniestros_integrados.csv"

print("=" * 75)
print("AUDITORIA DE DIRECCIONES GEOGRAFICAS")
print("=" * 75)

df = pd.read_csv(RUTA, low_memory=False)

direcciones = df["DIRECCION"].fillna("").astype(str).str.strip()

total = len(direcciones)
vacias = direcciones.eq("").sum()
unicas = direcciones.nunique()
repetidas = total - unicas

print("\n1. RESUMEN")
print("Total de registros:", total)
print("Direcciones únicas:", unicas)
print("Registros con direcciones repetidas:", repetidas)
print("Direcciones vacías:", vacias)

# Posibles valores genéricos
patrones_genericos = [
    r"^NO\s+REPORT",
    r"^SIN\s+INFORM",
    r"^DESCONOCID",
    r"^N/?A$",
    r"^NO\s+APLICA$",
    r"^S/?D$"
]

genericas = direcciones.str.upper().apply(
    lambda x: any(re.search(p, x) for p in patrones_genericos)
    if x else False
)

print("\n2. POSIBLES DIRECCIONES GENERICAS")
print("Cantidad:", genericas.sum())
print("Porcentaje:", round(genericas.mean() * 100, 2), "%")

# Posibles intersecciones
patrones_interseccion = [
    r"\bCON\b",
    r"\bCRUCE\b",
    r"\bESQUINA\b",
    r"\bINTERSECCION\b",
    r"\bCLL\b",
    r"\bCRA\b",
    r"\bCALLE\b",
    r"\bCARRERA\b"
]

texto = direcciones.str.upper()

posibles_intersecciones = texto.apply(
    lambda x: sum(bool(re.search(p, x)) for p in patrones_interseccion) >= 2
)

print("\n3. POSIBLES INTERSECCIONES")
print("Cantidad:", posibles_intersecciones.sum())
print(
    "Porcentaje:",
    round(posibles_intersecciones.mean() * 100, 2),
    "%"
)

# Longitud y patrones
longitudes = direcciones.str.len()

print("\n4. LONGITUD DE DIRECCIONES")
print("Longitud mínima:", longitudes.min())
print("Longitud media:", round(longitudes.mean(), 2))
print("Longitud máxima:", longitudes.max())

# Direcciones más frecuentes
print("\n5. 20 DIRECCIONES MAS FRECUENTES")
print(
    direcciones.value_counts().head(20).to_string()
)

# Muestra de direcciones
print("\n6. MUESTRA DE DIRECCIONES")
for direccion in direcciones.drop_duplicates().head(30):
    print("-", direccion)

# Guardar resumen
resumen = pd.DataFrame({
    "indicador": [
        "total_registros",
        "direcciones_unicas",
        "registros_repetidos",
        "direcciones_vacias",
        "posibles_genericas",
        "posibles_intersecciones",
        "longitud_media"
    ],
    "valor": [
        total,
        unicas,
        repetidas,
        vacias,
        int(genericas.sum()),
        int(posibles_intersecciones.sum()),
        round(longitudes.mean(), 2)
    ]
})

resumen.to_csv(
    "models/auditoria_direcciones_geograficas.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:")
print("models/auditoria_direcciones_geograficas.csv")

print("\nAuditoria terminada.")
