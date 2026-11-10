"""Compara V9D con referencias simples sobre la misma prueba temporal."""

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


BASE = Path(__file__).resolve().parents[2]
ARCHIVO_DATOS = BASE / "data" / "processed" / "siniestros_modelo.csv"
ARCHIVO_PREDICCIONES = BASE / "models" / "predicciones_v9d_sin_diseno.csv"
ARCHIVO_METRICAS = BASE / "models" / "comparacion_baselines_v9d_sin_diseno.json"
ARCHIVO_DETALLE = BASE / "models" / "predicciones_comparacion_baselines_v9d_sin_diseno.csv"
CLASE_FATAL = "Con Muertos"


def calcular_metricas(y_real, y_predicho, clases, clase_fatal=CLASE_FATAL):
    y_real = pd.Series(y_real).astype(str).reset_index(drop=True)
    y_predicho = pd.Series(y_predicho).astype(str).reset_index(drop=True)
    if len(y_real) != len(y_predicho):
        raise ValueError("Las etiquetas reales y predichas tienen longitudes distintas.")

    return {
        "accuracy": float(accuracy_score(y_real, y_predicho)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, y_predicho)),
        "f1_macro": float(
            f1_score(y_real, y_predicho, labels=clases, average="macro", zero_division=0)
        ),
        "precision_con_muertos": float(
            precision_score(
                y_real, y_predicho, labels=[clase_fatal], average="micro", zero_division=0
            )
        ),
        "recall_con_muertos": float(
            recall_score(
                y_real, y_predicho, labels=[clase_fatal], average="micro", zero_division=0
            )
        ),
        "f1_con_muertos": float(
            f1_score(
                y_real, y_predicho, labels=[clase_fatal], average="micro", zero_division=0
            )
        ),
    }


def comparar(train, test, predicciones_v9d):
    clases = sorted(set(train["GRAVEDAD"].astype(str)) | set(test["GRAVEDAD"].astype(str)))
    y_real = test["GRAVEDAD"].astype(str).reset_index(drop=True)
    pred_v9d = predicciones_v9d["GRAVEDAD_PREDICHA"].astype(str).reset_index(drop=True)
    real_guardado = predicciones_v9d["GRAVEDAD_REAL"].astype(str).reset_index(drop=True)

    if len(test) != len(predicciones_v9d):
        raise ValueError(
            f"Prueba temporal ({len(test)}) y predicciones V9D "
            f"({len(predicciones_v9d)}) tienen tamaños distintos."
        )
    if not y_real.equals(real_guardado):
        raise ValueError(
            "Las etiquetas reales de las predicciones V9D no coinciden "
            "fila por fila con la prueba temporal."
        )

    clase_mayoritaria = train["GRAVEDAD"].astype(str).mode().iloc[0]
    pred_mayoritaria = pd.Series(clase_mayoritaria, index=test.index).reset_index(drop=True)

    modas_localidad = train.groupby("CODIGO_LOCALIDAD")["GRAVEDAD"].agg(
        lambda valores: valores.astype(str).mode().iloc[0]
    )
    pred_localidad = (
        test["CODIGO_LOCALIDAD"].map(modas_localidad)
        .fillna(clase_mayoritaria)
        .astype(str)
        .reset_index(drop=True)
    )

    metricas = {
        "mayoria_global": {
            "clase_predicha": str(clase_mayoritaria),
            **calcular_metricas(y_real, pred_mayoritaria, clases),
        },
        "moda_por_localidad": {
            "fallback_localidad_desconocida": str(clase_mayoritaria),
            **calcular_metricas(y_real, pred_localidad, clases),
        },
        "V9D_sin_diseno": calcular_metricas(y_real, pred_v9d, clases),
    }

    detalle = test[["ANO", "CODIGO_LOCALIDAD"]].reset_index(drop=True).copy()
    detalle["GRAVEDAD_REAL"] = y_real
    detalle["PREDICCION_V9D"] = pred_v9d
    detalle["PREDICCION_MAYORIA_GLOBAL"] = pred_mayoritaria
    detalle["PREDICCION_MODA_LOCALIDAD"] = pred_localidad
    return metricas, detalle


def main():
    if not ARCHIVO_DATOS.exists() or not ARCHIVO_PREDICCIONES.exists():
        raise FileNotFoundError("Falta el dataset o el archivo de predicciones V9D.")

    datos = pd.read_csv(
        ARCHIVO_DATOS,
        usecols=["ANO", "CODIGO_LOCALIDAD", "GRAVEDAD"],
        low_memory=False,
    )
    datos["ANO"] = pd.to_numeric(datos["ANO"], errors="coerce")
    datos = datos.dropna(subset=["ANO", "GRAVEDAD"]).copy()
    datos["GRAVEDAD"] = datos["GRAVEDAD"].astype(str).str.strip()
    datos = datos[datos["GRAVEDAD"] != ""].copy()
    datos["CODIGO_LOCALIDAD"] = (
        datos["CODIGO_LOCALIDAD"]
        .astype("string")
        .fillna("DESCONOCIDO")
        .astype(str)
    )

    train = datos[datos["ANO"] <= 2018].copy()
    test = datos[datos["ANO"].between(2019, 2020)].copy()
    predicciones = pd.read_csv(ARCHIVO_PREDICCIONES, low_memory=False)
    metricas, detalle = comparar(train, test, predicciones)

    resultado = {
        "protocolo": "Entrenamiento hasta 2018; prueba temporal 2019-2020",
        "registros_entrenamiento_referencia": int(len(train)),
        "registros_prueba": int(len(test)),
        "clase_fatal": CLASE_FATAL,
        "modelos": metricas,
        "nota": (
            "V9D clasifica la gravedad entre siniestros registrados; "
            "no estima la ocurrencia de accidentes. Las referencias simples "
            "se calculan sobre la misma prueba temporal."
        ),
    }
    ARCHIVO_METRICAS.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    detalle.to_csv(ARCHIVO_DETALLE, index=False, encoding="utf-8-sig")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(f"Predicciones comparativas guardadas: {ARCHIVO_DETALLE}")


if __name__ == "__main__":
    main()
