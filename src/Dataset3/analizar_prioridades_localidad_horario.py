
from pathlib import Path
import pandas as pd

# =====================================================
# CONFIGURACIÓN
# =====================================================

RAIZ = Path(__file__).resolve().parents[2]

CSV = RAIZ / "data" / "processed" / "siniestros_modelo.csv"
SALIDA = RAIZ / "models"

SALIDA.mkdir(parents=True, exist_ok=True)

GRAVEDADES = [
    "Solo Daños",
    "Con Heridos",
    "Con Muertos",
]


def guardar(df, nombre):
    ruta = SALIDA / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"Generado: {ruta}")


def main():

    print("=" * 60)
    print("ANÁLISIS DE PRIORIDADES TERRITORIALES Y TEMPORALES")
    print("=" * 60)

    df = pd.read_csv(CSV)

    requeridas = [
        "CODIGO_LOCALIDAD",
        "GRAVEDAD",
        "ANO",
        "MES",
        "HORA_NUM",
        "DIA_SEMANA",
    ]

    faltantes = [
        c for c in requeridas if c not in df.columns
    ]

    if faltantes:
        print("Faltan columnas:", faltantes)
        return

    df = df.dropna(subset=requeridas).copy()

    df["HORA"] = (
        pd.to_numeric(df["HORA_NUM"], errors="coerce")
        .apply(lambda x: int(x) % 24 if pd.notna(x) else pd.NA)
    )

    # =================================================
    # 1. GRAVEDAD POR LOCALIDAD
    # =================================================

    print("\n1. GRAVEDAD POR LOCALIDAD")

    localidad = pd.crosstab(
        df["CODIGO_LOCALIDAD"],
        df["GRAVEDAD"]
    )

    for clase in GRAVEDADES:
        if clase not in localidad.columns:
            localidad[clase] = 0

    localidad = localidad[GRAVEDADES]

    localidad["TOTAL_SINIESTROS"] = localidad.sum(axis=1)

    localidad["PORCENTAJE_HERIDOS"] = (
        localidad["Con Heridos"]
        / localidad["TOTAL_SINIESTROS"] * 100
    ).round(2)

    localidad["PORCENTAJE_MUERTOS"] = (
        localidad["Con Muertos"]
        / localidad["TOTAL_SINIESTROS"] * 100
    ).round(2)

    localidad["PORCENTAJE_GRAVES"] = (
        (
            localidad["Con Heridos"]
            + localidad["Con Muertos"]
        )
        / localidad["TOTAL_SINIESTROS"] * 100
    ).round(2)

    localidad = localidad.reset_index()

    localidad = localidad.sort_values(
        "TOTAL_SINIESTROS",
        ascending=False
    )

    guardar(
        localidad,
        "prioridad_gravedad_por_localidad.csv"
    )

    print(
        localidad.head(10).to_string(index=False)
    )

    # =================================================
    # 2. GRAVEDAD POR HORA
    # =================================================

    print("\n2. GRAVEDAD POR HORA")

    hora = pd.crosstab(
        df["HORA"],
        df["GRAVEDAD"]
    )

    for clase in GRAVEDADES:
        if clase not in hora.columns:
            hora[clase] = 0

    hora = hora[GRAVEDADES]

    hora["TOTAL_SINIESTROS"] = hora.sum(axis=1)

    hora["PORCENTAJE_MUERTOS"] = (
        hora["Con Muertos"]
        / hora["TOTAL_SINIESTROS"] * 100
    ).round(2)

    hora["PORCENTAJE_GRAVES"] = (
        (
            hora["Con Heridos"]
            + hora["Con Muertos"]
        )
        / hora["TOTAL_SINIESTROS"] * 100
    ).round(2)

    hora = hora.reset_index().sort_values("HORA")

    guardar(
        hora,
        "prioridad_gravedad_por_hora.csv"
    )

    print(
        hora.sort_values(
            "TOTAL_SINIESTROS",
            ascending=False
        ).head(10).to_string(index=False)
    )

    # =================================================
    # 3. MATRIZ DE GRAVEDAD POR HORA
    # =================================================

    print("\n3. MATRIZ DE GRAVEDAD POR HORA")

    matriz_hora = pd.pivot_table(
        df,
        index="HORA",
        columns="GRAVEDAD",
        values="CODIGO_LOCALIDAD",
        aggfunc="count",
        fill_value=0
    )

    guardar(
        matriz_hora.reset_index(),
        "matriz_gravedad_hora.csv"
    )

    # =================================================
    # 4. GRAVEDAD POR DÍA DE SEMANA
    # =================================================

    print("\n4. GRAVEDAD POR CÓDIGO DE DÍA")

    dia = pd.crosstab(
        df["DIA_SEMANA"],
        df["GRAVEDAD"]
    )

    for clase in GRAVEDADES:
        if clase not in dia.columns:
            dia[clase] = 0

    dia = dia[GRAVEDADES]

    dia["TOTAL_SINIESTROS"] = dia.sum(axis=1)

    dia["PORCENTAJE_GRAVES"] = (
        (
            dia["Con Heridos"]
            + dia["Con Muertos"]
        )
        / dia["TOTAL_SINIESTROS"] * 100
    ).round(2)

    dia = dia.reset_index().sort_values("DIA_SEMANA")

    guardar(
        dia,
        "prioridad_gravedad_por_dia.csv"
    )

    print(dia.to_string(index=False))

    # =================================================
    # 5. EVOLUCIÓN ANUAL DE GRAVEDAD
    # =================================================

    print("\n5. EVOLUCIÓN ANUAL")

    anual = pd.crosstab(
        df["ANO"],
        df["GRAVEDAD"]
    )

    for clase in GRAVEDADES:
        if clase not in anual.columns:
            anual[clase] = 0

    anual = anual[GRAVEDADES]

    anual["TOTAL_SINIESTROS"] = anual.sum(axis=1)

    anual["PORCENTAJE_MUERTOS"] = (
        anual["Con Muertos"]
        / anual["TOTAL_SINIESTROS"] * 100
    ).round(2)

    anual = anual.reset_index().sort_values("ANO")

    guardar(
        anual,
        "evolucion_anual_gravedad.csv"
    )

    print(anual.to_string(index=False))

    # =================================================
    # 6. MATRIZ LOCALIDAD Y HORA
    # =================================================

    print("\n6. MATRIZ LOCALIDAD Y HORA")

    localidad_hora = (
        df.groupby(
            ["CODIGO_LOCALIDAD", "HORA"]
        )
        .size()
        .reset_index(name="TOTAL_SINIESTROS")
    )

    guardar(
        localidad_hora,
        "siniestros_localidad_hora.csv"
    )

    print(
        localidad_hora.sort_values(
            "TOTAL_SINIESTROS",
            ascending=False
        ).head(15).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("Archivos guardados en:", SALIDA)
    print("=" * 60)

    print(
        "\nIMPORTANTE: los porcentajes son distribuciones "
        "dentro de los siniestros registrados, no tasas "
        "de riesgo por población o volumen de tránsito."
    )


if __name__ == "__main__":
    main()
