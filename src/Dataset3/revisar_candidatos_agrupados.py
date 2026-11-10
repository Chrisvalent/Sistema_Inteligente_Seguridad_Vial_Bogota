import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

ARCHIVO = (
    BASE / "models" /
    "candidatos_agrupados_espacialmente.csv"
)

df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

print("=" * 65)
print("REVISIÓN DE CANDIDATOS AGRUPADOS")
print("=" * 65)

print("\nColumnas:")
for columna in df.columns:
    print("-", columna)

print("\nDimensiones:")
print("Filas:", len(df))
print("Columnas:", len(df.columns))

print("\nPrimeros cinco registros:")
print(df.head().to_string(index=False))