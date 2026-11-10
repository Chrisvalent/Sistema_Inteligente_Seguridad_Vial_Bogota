import pandas as pd

modelo = pd.read_csv("data/processed/siniestros_modelo.csv", low_memory=False)
integrado = pd.read_csv("data/processed/siniestros_integrados.csv", low_memory=False)

duplicados = modelo[modelo.duplicated(keep=False)].copy()

integrado["FECHA_DT"] = pd.to_datetime(
    integrado["FECHA"], dayfirst=True, errors="coerce"
)

duplicados["FECHA_DT"] = pd.to_datetime(
    dict(
        year=duplicados["ANO"],
        month=duplicados["MES"],
        day=duplicados["DIA"]
    ),
    errors="coerce"
)

integrado["HORA_TD"] = pd.to_timedelta(
    integrado["HORA"].astype(str), errors="coerce"
)

integrado["HORA_SEG"] = integrado["HORA_TD"].dt.total_seconds()

duplicados["HORA_SEG"] = (
    pd.to_numeric(duplicados["HORA_NUM"], errors="coerce") * 3600
)

integrado["CODIGO_LOCALIDAD"] = pd.to_numeric(
    integrado["CODIGO_LOCALIDAD"], errors="coerce"
)

duplicados["CODIGO_LOCALIDAD"] = pd.to_numeric(
    duplicados["CODIGO_LOCALIDAD"], errors="coerce"
)

resultados = []

for _, fila in duplicados.iterrows():

    candidatos = integrado[
        (integrado["FECHA_DT"] == fila["FECHA_DT"]) &
        (integrado["CODIGO_LOCALIDAD"] == fila["CODIGO_LOCALIDAD"]) &
        (integrado["CLASE"] == fila["CLASE"]) &
        (integrado["GRAVEDAD"] == fila["GRAVEDAD"])
    ].copy()

    candidatos["DIF_HORA_SEG"] = (
        candidatos["HORA_SEG"] - fila["HORA_SEG"]
    ).abs()

    candidatos = candidatos[candidatos["DIF_HORA_SEG"] <= 60]

    for _, candidato in candidatos.iterrows():
        resultados.append({
            "CODIGO_ACCIDENTE": candidato["CODIGO_ACCIDENTE"],
            "FECHA": candidato["FECHA"],
            "HORA": candidato["HORA"],
            "CODIGO_LOCALIDAD": candidato["CODIGO_LOCALIDAD"],
            "CLASE": candidato["CLASE"],
            "GRAVEDAD": candidato["GRAVEDAD"],
            "DIF_HORA_SEG": candidato["DIF_HORA_SEG"]
        })

resultado = pd.DataFrame(resultados)

print("=" * 70)
print("RESULTADO DE COMPARACION CON DATOS ORIGINALES")
print("=" * 70)

if not resultado.empty:
    print(resultado.to_string(index=False))
    print("\nCodigos unicos:", resultado["CODIGO_ACCIDENTE"].nunique())
    print("Registros candidatos:", len(resultado))
else:
    print("No se encontraron coincidencias.")

resultado.to_csv(
    "models/candidatos_duplicados_origen.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado: models/candidatos_duplicados_origen.csv")
