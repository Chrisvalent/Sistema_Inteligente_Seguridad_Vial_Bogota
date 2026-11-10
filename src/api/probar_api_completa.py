
from pathlib import Path
import json
import urllib.request
import urllib.error

import pandas as pd


RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "data" / "processed" / "siniestros_modelo.csv"
URL = "http://127.0.0.1:8000"

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


def solicitar(endpoint, datos=None):
    cuerpo = None
    cabeceras = {"Accept": "application/json"}

    if datos is not None:
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        cabeceras["Content-Type"] = "application/json"

    peticion = urllib.request.Request(
        URL + endpoint,
        data=cuerpo,
        headers=cabeceras,
        method="POST" if datos is not None else "GET",
    )

    with urllib.request.urlopen(peticion, timeout=60) as respuesta:
        return json.loads(respuesta.read().decode("utf-8"))


def valor_json(valor):
    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and valor.is_integer():
        return int(valor)
    return valor


def main():
    print("=" * 55)
    print("VALIDACIÓN TÉCNICA DE API Y POSTGRESQL")
    print("=" * 55)

    # 1. Verificar conexión
    try:
        salud = solicitar("/health")
        print("\n[OK] API y base de datos:", salud)
    except Exception as error:
        print("\n[ERROR] No se pudo consultar /health:", error)
        return

    # 2. Verificar el esquema del modelo
    try:
        esquema = solicitar("/schema")
        print("[OK] Variables esperadas:", esquema["cantidad_variables"])
    except Exception as error:
        print("[ERROR] Falló /schema:", error)
        return

    # 3. Leer cinco registros completos del CSV
    if not CSV.exists():
        print("[ERROR] No se encuentra el CSV:", CSV)
        return

    df = pd.read_csv(CSV)
    df = df.dropna(subset=VARIABLES)

    if len(df) < 5:
        print("[ERROR] No hay cinco registros completos para probar.")
        return

    muestra = df.head(5)
    ids_creados = []

    # 4. Enviar cinco solicitudes
    print("\nEnviando cinco solicitudes de prueba...")
    for numero, (_, fila) in enumerate(muestra.iterrows(), start=1):
        features = {
            campo: valor_json(fila[campo])
            for campo in VARIABLES
        }

        try:
            resultado = solicitar("/predict", {"features": features})
            registro_id = resultado.get("id_registro")

            if registro_id is None:
                print(f"[ERROR] Solicitud {numero}: no devolvió ID.")
                continue

            ids_creados.append(registro_id)
            print(
                f"[OK] Prueba {numero}: "
                f"ID={registro_id}, "
                f"predicción={resultado['prediccion']}"
            )

        except urllib.error.HTTPError as error:
            detalle = error.read().decode("utf-8", errors="replace")
            print(f"[ERROR] Prueba {numero}: HTTP {error.code} - {detalle}")

        except Exception as error:
            print(f"[ERROR] Prueba {numero}: {error}")

    # 5. Consultar los registros recientes
    print("\nConsultando registros recientes...")
    try:
        recientes = solicitar("/predicciones/recientes?limite=100")
        registros = recientes.get("predicciones", [])
        ids_guardados = {r["id"] for r in registros}

        confirmados = sum(
            identificador in ids_guardados
            for identificador in ids_creados
        )

        print("Solicitudes guardadas según la API:", len(ids_creados))
        print("IDs encontrados en la consulta reciente:", confirmados)

        if confirmados == len(ids_creados) and ids_creados:
            print("\nRESULTADO: PRUEBA TÉCNICA SUPERADA.")
        else:
            print("\nRESULTADO: REVISAR REGISTROS NO CONFIRMADOS.")

    except Exception as error:
        print("[ERROR] No se pudieron consultar los registros:", error)

    print("\nNota: se utilizaron registros históricos para probar la API.")
    print("No son predicciones independientes ni una validación del modelo.")


if __name__ == "__main__":
    main()
