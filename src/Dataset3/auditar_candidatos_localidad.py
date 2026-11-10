
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

ARCHIVO_CANDIDATOS = (
    BASE / "models" / "candidatos_filtrados_localidad.csv"
)

ARCHIVO_RESUMEN = (
    BASE / "models" / "resumen_filtrado_localidad.csv"
)

SALIDA = (
    BASE / "models" / "auditoria_candidatos_compatibles.csv"
)


def main():

    print("=" * 70)
    print("AUDITORÍA DE CANDIDATOS COMPATIBLES")
    print("=" * 70)

    candidatos = pd.read_csv(
        ARCHIVO_CANDIDATOS,
        encoding="utf-8-sig"
    )

    resumen = pd.read_csv(
        ARCHIVO_RESUMEN,
        encoding="utf-8-sig"
    )

    compatibles = candidatos[
        candidatos["DENTRO_LOCALIDAD"] == True
    ].copy()

    # Resumen de candidatos por dirección
    conteos = (
        compatibles.groupby("DIRECCION")
        .size()
        .rename("NUM_CANDIDATOS")
    )

    # Agregar número de candidatos al inventario
    resultado = resumen.merge(
        conteos,
        on="DIRECCION",
        how="left"
    )

    resultado["NUM_CANDIDATOS"] = (
        resultado["NUM_CANDIDATOS"]
        .fillna(0)
        .astype(int)
    )

    # Solo direcciones con candidatos compatibles
    direcciones = resultado[
        resultado["NUM_CANDIDATOS"] > 0
    ].copy()

    # Incorporar las coordenadas de cada candidato
    detalle = compatibles.merge(
        direcciones[
            [
                "DIRECCION",
                "ESTADO_LOCALIDAD",
                "NUM_CANDIDATOS"
            ]
        ],
        on="DIRECCION",
        how="left"
    )

    columnas = [
        "DIRECCION",
        "CODIGO_LOCALIDAD",
        "NOMBRE_LOCALIDAD",
        "VIA_1",
        "VIA_2",
        "CANDIDATO",
        "LONGITUD",
        "LATITUD",
        "NUM_CANDIDATOS",
        "ESTADO_LOCALIDAD",
        "VALIDADO"
    ]

    detalle = detalle[columnas]

    detalle.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDirecciones con candidatos:", len(direcciones))

    print(
        "Direcciones con un candidato:",
        (direcciones["NUM_CANDIDATOS"] == 1).sum()
    )

    print(
        "Direcciones con varios candidatos:",
        (direcciones["NUM_CANDIDATOS"] > 1).sum()
    )

    print(
        "\nTotal de coordenadas compatibles:",
        len(detalle)
    )

    print("\nPrimeras 30 coordenadas:")

    print(
        detalle.head(30).to_string(index=False)
    )

    print("\nArchivo generado:")
    print(SALIDA)

    print("\nAuditoría terminada.")


if __name__ == "__main__":
    main()
