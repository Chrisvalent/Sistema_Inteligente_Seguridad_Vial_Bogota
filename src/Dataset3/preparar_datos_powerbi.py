
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
MODELS = RAIZ / "models"

LOCALIDADES = {
    1: "Usaquén",
    2: "Chapinero",
    3: "Santa Fe",
    4: "San Cristóbal",
    5: "Usme",
    6: "Tunjuelito",
    7: "Bosa",
    8: "Kennedy",
    9: "Fontibón",
    10: "Engativá",
    11: "Suba",
    12: "Barrios Unidos",
    13: "Teusaquillo",
    14: "Los Mártires",
    15: "Antonio Nariño",
    16: "Puente Aranda",
    17: "La Candelaria",
    18: "Rafael Uribe Uribe",
    19: "Ciudad Bolívar",
    20: "Sumapaz",
}

# Convención de lunes=0 a domingo=6.
# Es consistente con el registro del 01/01/2015,
# que corresponde a jueves y aparece con código 3.
DIAS = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}


def cargar(nombre):
    ruta = MODELS / nombre

    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")

    return pd.read_csv(ruta)


def guardar(df, nombre):
    ruta = MODELS / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"Generado: {ruta}")


def main():

    print("=" * 60)
    print("PREPARACIÓN DE DATOS PARA POWER BI")
    print("=" * 60)

    # 1. Localidades y gravedad
    localidad = cargar(
        "prioridad_gravedad_por_localidad.csv"
    )

    localidad["NOMBRE_LOCALIDAD"] = (
        pd.to_numeric(
            localidad["CODIGO_LOCALIDAD"],
            errors="coerce"
        )
        .map(LOCALIDADES)
    )

    guardar(
        localidad,
        "powerbi_gravedad_localidad.csv"
    )

    # 2. Horarios y gravedad
    hora = cargar(
        "prioridad_gravedad_por_hora.csv"
    )

    hora["FRANJA_HORA"] = hora["HORA"].apply(
        lambda h: f"{int(h):02d}:00"
    )

    guardar(
        hora,
        "powerbi_gravedad_hora.csv"
    )

    # 3. Días y gravedad
    dia = cargar(
        "prioridad_gravedad_por_dia.csv"
    )

    dia["NOMBRE_DIA"] = (
        pd.to_numeric(
            dia["DIA_SEMANA"],
            errors="coerce"
        )
        .map(DIAS)
    )

    guardar(
        dia,
        "powerbi_gravedad_dia.csv"
    )

    # 4. Localidad y hora
    localidad_hora = cargar(
        "siniestros_localidad_hora.csv"
    )

    localidad_hora["NOMBRE_LOCALIDAD"] = (
        pd.to_numeric(
            localidad_hora["CODIGO_LOCALIDAD"],
            errors="coerce"
        )
        .map(LOCALIDADES)
    )

    localidad_hora["FRANJA_HORA"] = (
        localidad_hora["HORA"].apply(
            lambda h: f"{int(h):02d}:00"
        )
    )

    guardar(
        localidad_hora,
        "powerbi_localidad_hora.csv"
    )

    # 5. Evolución anual
    anual = cargar("evolucion_anual_gravedad.csv")

    guardar(
        anual,
        "powerbi_evolucion_anual.csv"
    )

    print("\nPREPARACIÓN FINALIZADA.")
    print("Archivos disponibles en:", MODELS)


if __name__ == "__main__":
    main()
