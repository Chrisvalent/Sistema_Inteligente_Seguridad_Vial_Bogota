
import os
import pandas as pd

RUTA_EXCEL = (
    "data/raw/"
    "siniestros_viales_consolidados_bogota_dc.xlsx"
)

RUTA_INTEGRADOS = "data/processed/siniestros_integrados.csv"
RUTA_SALIDA = "models/diagnostico_viabilidad_preventiva.csv"

print("=" * 80)
print("DIAGNOSTICO DE VIABILIDAD DEL MODELO PREVENTIVO")
print("=" * 80)

# Palabras clave para localizar información útil
grupos = {
    "Temporal": [
        "FECHA", "HORA", "ANO", "MES", "DIA"
    ],
    "Geografica": [
        "LOCALIDAD", "BARRIO", "DIRECCION",
        "LATITUD", "LONGITUD", "COORDENADA",
        "LAT", "LON"
    ],
    "Infraestructura vial": [
        "VIA", "CALZADA", "CARRIL", "INTERSECCION",
        "DISENO", "SENAL", "SEMAFORO", "CICLORUTA",
        "PAVIMENTO", "ILUMINACION", "VELOCIDAD"
    ],
    "Flujo y exposicion": [
        "FLUJO", "AFORO", "TRANSITO", "TRAFICO",
        "VOLUMEN", "VEHICULOS_DIA", "TPDA",
        "MOVILIDAD", "CONTEO"
    ],
    "Meteorologica y ambiental": [
        "CLIMA", "LLUVIA", "PRECIPITACION",
        "TEMPERATURA", "HUMEDAD", "VIENTO",
        "VISIBILIDAD", "HORA_SOLAR"
    ]
}

resultados = []


def buscar_grupo(nombre_columna):
    nombre = str(nombre_columna).upper()

    for grupo, palabras in grupos.items():
        if any(p in nombre for p in palabras):
            return grupo

    return None


# ---------------------------------------------------------
# 1. Revisar archivo original de Excel
# ---------------------------------------------------------

print("\n1. REVISANDO ARCHIVO ORIGINAL DE EXCEL")
print("-" * 80)

if not os.path.exists(RUTA_EXCEL):
    print("No se encontró:", RUTA_EXCEL)
else:
    excel = pd.ExcelFile(RUTA_EXCEL)

    print("Hojas encontradas:", excel.sheet_names)

    for hoja in excel.sheet_names:
        print(f"\nHOJA: {hoja}")

        encabezados = pd.read_excel(
            RUTA_EXCEL,
            sheet_name=hoja,
            nrows=0
        ).columns.tolist()

        print("Total de columnas:", len(encabezados))

        for columna in encabezados:
            grupo = buscar_grupo(columna)

            if grupo:
                print(f"  [{grupo}] {columna}")

                resultados.append({
                    "fuente": "Excel original",
                    "hoja": hoja,
                    "variable": str(columna),
                    "grupo": grupo,
                    "disponibilidad": "Por verificar",
                    "observacion": (
                        "Existe como columna en la fuente original; "
                        "no demuestra disponibilidad antes del accidente."
                    )
                })


# ---------------------------------------------------------
# 2. Revisar archivo integrado
# ---------------------------------------------------------

print("\n\n2. REVISANDO DATASET INTEGRADO")
print("-" * 80)

if not os.path.exists(RUTA_INTEGRADOS):
    print("No se encontró:", RUTA_INTEGRADOS)

else:
    df = pd.read_csv(
        RUTA_INTEGRADOS,
        low_memory=False
    )

    print("Registros:", len(df))
    print("Columnas:", len(df.columns))

    columnas_interes = [
        "FECHA",
        "HORA",
        "CODIGO_LOCALIDAD",
        "DISENO_LUGAR",
        "DIRECCION",
        "GRAVEDAD"
    ]

    print("\nDisponibilidad de campos principales:")

    for columna in columnas_interes:
        if columna not in df.columns:
            print(f"\n{columna}: NO ENCONTRADA")
            continue

        faltantes = df[columna].isna().sum()
        porcentaje = faltantes / len(df) * 100

        print(f"\n{columna}")
        print("  Registros no nulos:", len(df) - faltantes)
        print("  Valores faltantes:", faltantes)
        print("  Porcentaje faltante:", round(porcentaje, 2), "%")
        print("  Valores distintos:", df[columna].nunique())

        if columna != "GRAVEDAD":
            resultados.append({
                "fuente": "Dataset integrado",
                "hoja": "siniestros_integrados",
                "variable": columna,
                "grupo": buscar_grupo(columna) or "Otra",
                "disponibilidad": "Por verificar",
                "observacion": (
                    f"Faltantes: {porcentaje:.2f} %. "
                    "La existencia del campo no demuestra "
                    "disponibilidad previa al accidente."
                )
            })

    # Buscar campos adicionales por nombre
    print("\nCampos adicionales de interés:")

    for columna in df.columns:
        grupo = buscar_grupo(columna)

        if grupo:
            print(f"  [{grupo}] {columna}")

    # Revisar rango temporal
    if "FECHA" in df.columns:
        fechas = pd.to_datetime(
            df["FECHA"],
            errors="coerce",
            dayfirst=True
        )

        print("\nPeriodo de los registros:")
        print("  Fecha inicial:", fechas.min())
        print("  Fecha final:", fechas.max())


# ---------------------------------------------------------
# 3. Resumen de campos identificados
# ---------------------------------------------------------

print("\n\n3. RESUMEN DE VARIABLES IDENTIFICADAS")
print("-" * 80)

if resultados:
    resultado = pd.DataFrame(resultados)

    resultado = resultado.drop_duplicates(
        subset=["fuente", "hoja", "variable"]
    )

    print(
        resultado.groupby(
            ["fuente", "grupo"]
        ).size().to_string()
    )

    os.makedirs("models", exist_ok=True)

    resultado.to_csv(
        RUTA_SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArchivo generado:", RUTA_SALIDA)

else:
    print("No se identificaron campos para el diagnóstico.")

print("\n" + "=" * 80)
print("DIAGNOSTICO TERMINADO")
print("=" * 80)

print(
    "\nIMPORTANTE: encontrar una columna en el archivo "
    "no demuestra que su información estuviera disponible "
    "antes del accidente."
)
