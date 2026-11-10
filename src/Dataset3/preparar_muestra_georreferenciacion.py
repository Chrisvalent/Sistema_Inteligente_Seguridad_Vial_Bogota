
import pandas as pd
import os

RUTA = "data/processed/siniestros_integrados.csv"
SALIDA = "data/processed/muestra_georreferenciacion.csv"

print("=" * 70)
print("PREPARACION DE MUESTRA PARA GEORREFERENCIACION")
print("=" * 70)

df = pd.read_csv(RUTA, low_memory=False)

# Agrupar por dirección y localidad
resumen = (
    df.groupby(
        ["DIRECCION", "CODIGO_LOCALIDAD"],
        dropna=False
    )
    .agg(
        cantidad_accidentes=("CODIGO_ACCIDENTE", "count"),
        fecha_minima=("FECHA", "min"),
        fecha_maxima=("FECHA", "max")
    )
    .reset_index()
)

# Ordenar por frecuencia histórica
resumen = resumen.sort_values(
    "cantidad_accidentes",
    ascending=False
)

# Seleccionar las 100 combinaciones más frecuentes
muestra = resumen.head(100).copy()

# Agregar columna vacía para futuras coordenadas
muestra["LATITUD"] = pd.NA
muestra["LONGITUD"] = pd.NA
muestra["ESTADO_GEORREFERENCIACION"] = "PENDIENTE"

os.makedirs("data/processed", exist_ok=True)

muestra.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\nRegistros históricos:", len(df))
print("Combinaciones dirección-localidad:", len(resumen))
print("Muestra seleccionada:", len(muestra))

print("\nPrimeras 20 direcciones:")
print(
    muestra[
        ["DIRECCION", "CODIGO_LOCALIDAD", "cantidad_accidentes"]
    ].head(20).to_string(index=False)
)

print("\nArchivo generado:", SALIDA)
print("\nProceso terminado.")
