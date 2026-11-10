
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


def convertir(valor):
    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and valor.is_integer():
        return int(valor)
    return valor


def enviar(endpoint, datos):
    cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
    peticion = urllib.request.Request(
        URL + endpoint,
        data=cuerpo,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(peticion, timeout=60) as respuesta:
            return respuesta.status, json.loads(
                respuesta.read().decode("utf-8")
            )
    except urllib.error.HTTPError as error:
        texto = error.read().decode("utf-8", errors="replace")
        try:
            contenido = json.loads(texto)
        except json.JSONDecodeError:
            contenido = {"detalle": texto}
        return error.code, contenido


def obtener_registros():
    peticion = urllib.request.Request(URL + "/predicciones/recientes?limite=100")
    with urllib.request.urlopen(peticion, timeout=30) as respuesta:
        datos = json.loads(respuesta.read().decode("utf-8"))
    return datos["predicciones"]


def main():
    print("=" * 55)
    print("PRUEBAS DE ROBUSTEZ DE LA API")
    print("=" * 55)

    # Obtener una fila real para construir la solicitud válida.
    if not CSV.exists():
        print("[ERROR] No se encontró el CSV:", CSV)
        return

    df = pd.read_csv(CSV, nrows=10)
    df = df.dropna(subset=VARIABLES)

    if df.empty:
        print("[ERROR] No hay filas completas en el CSV.")
        return

    fila = df.iloc[0]
    features = {campo: convertir(fila[campo]) for campo in VARIABLES}

    # Registrar el número de registros antes de probar.
    antes = obtener_registros()
    ids_antes = {r["id"] for r in antes}
    print(f"\nRegistros visibles antes de las pruebas: {len(ids_antes)}")

    resultados = []

    # Prueba 1: solicitud sin variables.
    codigo, respuesta = enviar("/predict", {"features": {}})
    ok = codigo == 422
    resultados.append(ok)
    print(f"\n1. Variables faltantes: HTTP {codigo} - "
          f"{'OK' if ok else 'REVISAR'}")
    if not ok:
        print(respuesta)

    # Prueba 2: variables adicionales.
    features_extra = dict(features)
    features_extra["VARIABLE_INVENTADA"] = 123
    codigo, respuesta = enviar(
        "/predict",
        {"features": features_extra},
    )
    ok = codigo == 422
    resultados.append(ok)
    print(f"\n2. Variable adicional: HTTP {codigo} - "
          f"{'OK' if ok else 'REVISAR'}")
    if not ok:
        print(respuesta)

    # Prueba 3: solicitud válida.
    codigo, respuesta = enviar(
        "/predict",
        {"features": features},
    )
    ok = codigo == 200 and "id_registro" in respuesta
    resultados.append(ok)
    print(f"\n3. Solicitud válida: HTTP {codigo} - "
          f"{'OK' if ok else 'REVISAR'}")
    if ok:
        print("   ID generado:", respuesta["id_registro"])
        print("   Predicción:", respuesta["prediccion"])
    else:
        print(respuesta)

    # Verificar qué registros nuevos aparecen en la consulta reciente.
    despues = obtener_registros()
    ids_despues = {r["id"] for r in despues}
    nuevos = ids_despues - ids_antes

    print("\n--- Verificación de registros ---")
    print("Nuevos IDs visibles:", sorted(nuevos))
    print("Se esperaba un registro nuevo para la solicitud válida.")

    # Las dos solicitudes inválidas no deben generar registros.
    # La solicitud válida sí debe crear exactamente uno.
    if len(nuevos) == 1 and ok:
        print("[OK] Se observó un solo registro nuevo.")
    else:
        print(
            "[AVISO] La consulta reciente no permite confirmar "
            "exactamente un registro nuevo. Si existen más de 100 "
            "registros, esta comprobación puede ser incompleta."
        )

    print("\n--- Resumen ---")
    print(f"Pruebas HTTP superadas: {sum(resultados)}/3")

    if all(resultados):
        print("RESULTADO: RESPUESTAS HTTP ESPERADAS.")
    else:
        print("RESULTADO: HAY PRUEBAS QUE REQUIEREN REVISIÓN.")

    print(
        "\nNota: este script valida respuestas HTTP y una comprobación "
        "básica del registro. No evalúa la precisión del modelo."
    )


if __name__ == "__main__":
    main()
