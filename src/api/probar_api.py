
from pathlib import Path
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "data" / "processed" / "siniestros_modelo.csv"
URL = "http://127.0.0.1:8000/predict"

VARIABLES = [
    "CLASE", "CHOQUE", "OBJETO_FIJO", "CODIGO_LOCALIDAD",
    "DISENO_LUGAR", "FRANJA_HORARIA", "ANO", "MES", "DIA",
    "DIA_SEMANA", "ES_FIN_DE_SEMANA", "HORA_NUM",
    "NUM_ACTORES", "NUM_CONDUCTORES", "NUM_PEATONES",
    "NUM_VEHICULOS", "NUM_VEHICULOS_EN_FUGA",
    "NUM_CLASES_VEHICULO", "NUM_SERVICIOS", "NUM_MODALIDADES",
    "NUM_AUTOMOVIL", "NUM_BUS", "NUM_BUSETA",
    "NUM_CAMION_FURGON", "NUM_CAMIONETA", "NUM_CAMPERO",
    "NUM_MICROBUS", "NUM_TRACTOCAMION", "NUM_VOLQUETA",
    "NUM_MOTOCICLETA", "NUM_BICICLETA", "NUM_MOTOCARRO",
    "NUM_BICITAXI", "NUM_MOTOS_TOTAL",
]


def normalizar(valor):
    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {CSV}")

    df = pd.read_csv(CSV)

    faltantes = [c for c in VARIABLES if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    # Buscar una fila completa para evitar valores ausentes en esta primera prueba.
    filas = df.dropna(subset=VARIABLES)

    if filas.empty:
        raise ValueError(
            "No hay registros completos para las 34 variables del modelo."
        )

    fila = filas.iloc[0]
    features = {
        nombre: normalizar(fila[nombre])
        for nombre in VARIABLES
    }

    payload = {"features": features}
    cuerpo = json.dumps(payload, allow_nan=False).encode("utf-8")

    solicitud = Request(
        URL,
        data=cuerpo,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("Enviando un registro real a la API...")
    print(f"CSV: {CSV}")
    print(f"Variables enviadas: {len(features)}")
    print(f"GRAVEDAD real del registro: {fila.get('GRAVEDAD', 'No disponible')}")

    try:
        with urlopen(solicitud, timeout=120) as respuesta:
            resultado = json.loads(respuesta.read().decode("utf-8"))
            print("\nRespuesta de la API:")
            print(json.dumps(resultado, indent=2, ensure_ascii=False))

    except HTTPError as error:
        print(f"\nError HTTP: {error.code}")
        print(error.read().decode("utf-8", errors="replace"))

    except URLError as error:
        print(f"\nNo fue posible conectar con la API: {error}")


if __name__ == "__main__":
    main()
