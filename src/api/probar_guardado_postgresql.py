"""Prueba manual de integración V8B con la API y PostgreSQL.

Crea un único registro marcado en `predicciones`; no borra ni cambia filas previas.
Ejecutar solo cuando se quiera verificar una inserción real.
"""
from datetime import datetime
import json
from pathlib import Path
import urllib.error
import urllib.request

import pandas as pd


RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "data" / "processed" / "siniestros_modelo.csv"
URL = "http://127.0.0.1:8000"


def solicitar(path, payload=None):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        URL + path,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"La API devolvió HTTP {error.code}: {detail}") from error


def convertir(valor):
    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and valor.is_integer():
        return int(valor)
    return valor


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {CSV}")

    estado = solicitar("/health")
    if estado.get("base_datos") != "conectada":
        raise RuntimeError("La API no informa conexión con PostgreSQL.")

    schema = solicitar("/schema")
    variables = schema["variables_requeridas"]
    df = pd.read_csv(CSV, usecols=variables, nrows=300)
    filas = df.dropna(subset=variables)
    if filas.empty:
        raise RuntimeError("No hay una fila completa para probar.")

    features = {nombre: convertir(filas.iloc[0][nombre]) for nombre in variables}
    marca = "__PRUEBA_TECNICA_API_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "__"
    if "CLASE" in features:
        features["CLASE"] = marca

    resultado = solicitar("/predict", {"features": features})
    if resultado.get("modelo") != "V8B" or "id_registro" not in resultado:
        raise RuntimeError("La API no devolvió una predicción V8B persistida.")
    if abs(sum(resultado["puntuaciones"].values()) - 1.0) > 1e-6:
        raise RuntimeError("Las puntuaciones no suman 1.")

    recientes = solicitar("/predicciones/recientes?limite=1")
    if not recientes["predicciones"] or recientes["predicciones"][0]["id"] != resultado["id_registro"]:
        raise RuntimeError("No se encontró el registro recién guardado.")

    print(json.dumps({
        "api": "conectada",
        "base_datos": "conectada",
        "modelo": resultado["modelo"],
        "id_prueba": resultado["id_registro"],
        "marca_en_variable_clase": marca if "CLASE" in features else None,
        "prediccion": resultado["prediccion"],
        "puntuaciones_suman_1": True,
        "persistencia_confirmada": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
