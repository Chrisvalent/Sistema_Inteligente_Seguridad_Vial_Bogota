import pandas as pd

ruta = "data/processed/siniestros_modelo.csv"

df = pd.read_csv(ruta, low_memory=False)

duplicados = df[df.duplicated(keep=False)].copy()

print("=" * 70)
print("AUDITORÍA DE LOS 18 DUPLICADOS")
print("=" * 70)

print("\nCantidad de filas involucradas:", len(duplicados))

print("\nDistribución de gravedad:")
print(duplicados["GRAVEDAD"].value_counts(dropna=False))

print("\nPrimeros registros duplicados:")
print(duplicados[["ANO", "MES", "DIA", "HORA_NUM",
                  "CODIGO_LOCALIDAD", "CLASE", "GRAVEDAD"]].to_string(index=False))

print("\nCantidad de grupos duplicados:")
print(duplicados.groupby(list(df.columns), dropna=False).ngroups)

duplicados.to_csv(
    "models/auditoria_duplicados_v8b.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado: models/auditoria_duplicados_v8b.csv")
