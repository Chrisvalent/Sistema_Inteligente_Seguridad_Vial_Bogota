
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "siniestros_modelo.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    necesarias = ["DISENO_LUGAR", "GRAVEDAD"]
    faltantes = [c for c in necesarias if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    diseno = df["DISENO_LUGAR"].fillna("SIN DATO").astype(str).str.strip()
    gravedad = df["GRAVEDAD"].fillna("SIN DATO").astype(str).str.strip()

    print("=" * 65)
    print("ANÁLISIS DE DISENO_LUGAR")
    print("=" * 65)

    print("\nValores más frecuentes:")
    print(diseno.value_counts(dropna=False).head(30).to_string())

    print("\nPorcentaje de valores faltantes:")
    print(f"{df['DISENO_LUGAR'].isna().mean() * 100:.2f}%")

    print("\nDistribución de gravedad por diseño del lugar (% por fila):")
    tabla = pd.crosstab(
        diseno,
        gravedad,
        normalize="index"
    ) * 100

    print(tabla.round(2).to_string())

    salida = ROOT / "models"
    salida.mkdir(parents=True, exist_ok=True)

    tabla.to_csv(
        salida / "distribucion_gravedad_por_diseno.csv",
        encoding="utf-8-sig",
    )

    print("\nTabla guardada en:")
    print(salida / "distribucion_gravedad_por_diseno.csv")


if __name__ == "__main__":
    main()
