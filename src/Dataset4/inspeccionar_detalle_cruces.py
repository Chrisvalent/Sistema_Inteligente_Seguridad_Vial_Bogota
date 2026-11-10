
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

ARCHIVO = (
    BASE
    / "models"
    / "detalle_cruces_geometricos_50_grupos.csv"
)

if not ARCHIVO.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo: {ARCHIVO}"
    )

df = pd.read_csv(
    ARCHIVO,
    encoding="utf-8-sig"
)

print("\n========================================")
print(" INSPECCION DEL DETALLE DE CRUCES")
print("========================================")

print(f"\nTotal de registros: {len(df)}")

print("\nCOLUMNAS DISPONIBLES:")
for columna in df.columns:
    print(f" - {columna}")

print("\nPRIMEROS 15 REGISTROS:")
print(df.head(15).to_string(index=False))

print("\nCANTIDAD DE VALORES NULOS:")
print(df.isna().sum().to_string())

print("\nREGISTROS DUPLICADOS EXACTOS:")
print(df.duplicated().sum())

# Buscar columnas de dirección y grupo
columnas = {
    str(c).strip().upper(): c
    for c in df.columns
}

col_direccion = columnas.get("DIRECCION")
col_grupo = columnas.get("GRUPO_ESPACIAL")

if col_direccion:
    print("\nCRUCES POR DIRECCION:")

    conteo = (
        df.groupby(col_direccion)
        .size()
        .sort_values(ascending=False)
    )

    print(conteo.head(20).to_string())

if col_direccion and col_grupo:
    print("\nCRUCES POR DIRECCION Y GRUPO:")

    conteo_grupos = (
        df.groupby([col_direccion, col_grupo])
        .size()
        .reset_index(name="NUMERO_CRUCES")
        .sort_values(
            "NUMERO_CRUCES",
            ascending=False
        )
    )

    print(
        conteo_grupos.head(30).to_string(index=False)
    )

print("\n========================================")
print(" INSPECCION TERMINADA")
print("========================================")
