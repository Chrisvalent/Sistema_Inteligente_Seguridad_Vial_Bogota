import pandas as pd
import os

# Carpeta donde están los datos originales
ruta_raw = "data/raw"

# Buscar archivos Excel
archivos = [
    f for f in os.listdir(ruta_raw)
    if f.lower().endswith((".xlsx", ".xls"))
]

print("Archivos Excel encontrados:")

for archivo in archivos:
    print("-", archivo)

if not archivos:
    print("\nNo se encontró ningún archivo Excel en data/raw")
    exit()

# Usar el archivo de siniestros
archivo = "Historico_siniestros_bogota.xlsx"
ruta = os.path.join(ruta_raw, archivo)

print(f"\nCargando: {archivo}")

# Leer Excel
df = pd.read_excel(ruta)

print("\n===== INFORMACIÓN DEL DATASET =====")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\n===== COLUMNAS =====")

for columna in df.columns:
    print("-", columna)

print("\n===== PRIMERAS 5 FILAS =====")
print(df.head().to_string())

print("\n===== TIPOS DE DATOS =====")
print(df.dtypes)

print("\n===== DATOS FALTANTES =====")
print(df.isnull().sum())

print("\n===== DUPLICADOS =====")
print("Duplicados:", df.duplicated().sum())