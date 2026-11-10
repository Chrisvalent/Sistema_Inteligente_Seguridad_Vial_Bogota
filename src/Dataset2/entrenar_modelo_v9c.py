
"""
V9C - Modelo preventivo de gravedad de siniestros viales.

Entrenamiento: ANO <= 2018.
Prueba temporal: ANO entre 2019 y 2020.

Variables preventivas:
ANO, MES, DIA, DIA_SEMANA, ES_FIN_DE_SEMANA,
HORA_NUM y CODIGO_LOCALIDAD.

Se excluyen CLASE, CHOQUE, actores, vehículos,
hipótesis y causas porque pueden contener información
posterior al siniestro.
"""

from pathlib import Path
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "siniestros_modelo.csv"
)


def buscar_columna(df, opciones):
    columnas = {str(c).strip().upper(): c for c in df.columns}

    for opcion in opciones:
        if opcion.upper() in columnas:
            return columnas[opcion.upper()]

    return None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}\n"
            "Ajusta DATA_PATH a la ubicación real del CSV."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    objetivo = buscar_columna(df, ["GRAVEDAD"])
    anio = buscar_columna(df, ["ANO", "ANIO", "AÑO"])
    localidad = buscar_columna(
        df, ["CODIGO_LOCALIDAD", "LOCALIDAD", "COD_LOCALIDAD"]
    )

    columnas_fecha = {
        "MES": buscar_columna(df, ["MES"]),
        "DIA": buscar_columna(df, ["DIA", "DÍA"]),
        "DIA_SEMANA": buscar_columna(
            df, ["DIA_SEMANA", "DÍA_SEMANA"]
        ),
        "ES_FIN_DE_SEMANA": buscar_columna(
            df, ["ES_FIN_DE_SEMANA"]
        ),
        "HORA_NUM": buscar_columna(df, ["HORA_NUM"]),
    }

    faltantes = []

    if objetivo is None:
        faltantes.append("GRAVEDAD")
    if anio is None:
        faltantes.append("ANO")
    if localidad is None:
        faltantes.append("CODIGO_LOCALIDAD o LOCALIDAD")

    faltantes.extend(
        nombre
        for nombre, columna in columnas_fecha.items()
        if columna is None
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias: {faltantes}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Convertir variables numéricas.
    variables_numericas = [
        anio,
        columnas_fecha["MES"],
        columnas_fecha["DIA"],
        columnas_fecha["DIA_SEMANA"],
        columnas_fecha["ES_FIN_DE_SEMANA"],
        columnas_fecha["HORA_NUM"],
    ]

    for columna in variables_numericas:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df[objetivo] = df[objetivo].astype("string").str.strip()

    df = df.dropna(subset=[anio, objetivo]).copy()
    df = df[df[objetivo] != ""].copy()

    # División cronológica: no mezclar años entre entrenamiento y prueba.
    train = df[df[anio] <= 2018].copy()
    test = df[(df[anio] >= 2019) & (df[anio] <= 2020)].copy()

    if train.empty or test.empty:
        raise ValueError(
            "La división temporal quedó vacía. "
            f"Años encontrados: {sorted(df[anio].dropna().unique())}"
        )

    numericas = [
        anio,
        columnas_fecha["MES"],
        columnas_fecha["DIA"],
        columnas_fecha["DIA_SEMANA"],
        columnas_fecha["ES_FIN_DE_SEMANA"],
        columnas_fecha["HORA_NUM"],
    ]
    categoricas = [localidad]

    for columna in categoricas:
        train[columna] = (
            train[columna].astype("string")
            .fillna("DESCONOCIDO").astype(str)
        )
        test[columna] = (
            test[columna].astype("string")
            .fillna("DESCONOCIDO").astype(str)
        )

    X_train = train[numericas + categoricas]
    X_test = test[numericas + categoricas]
    y_train = train[objetivo].astype(str)
    y_test = test[objetivo].astype(str)

    preprocesamiento = ColumnTransformer([
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputador", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ),
    ])

    modelo = Pipeline([
        ("preprocesamiento", preprocesamiento),
        (
            "clasificador",
            RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("=" * 65)
    print("MODELO V9C PREVENTIVO")
    print("=" * 65)
    print(f"Dataset: {DATA_PATH}")
    print(f"Registros de entrenamiento (hasta 2018): {len(train):,}")
    print(f"Registros de prueba (2019-2020): {len(test):,}")
    print("Variables:", numericas + categoricas)

    print("\nEntrenando...")
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    balanced = balanced_accuracy_score(y_test, pred)
    f1_macro = f1_score(
        y_test, pred, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        y_test, pred, average="weighted", zero_division=0
    )

    etiquetas = sorted(set(y_train.unique()) | set(y_test.unique()))
    matriz = confusion_matrix(y_test, pred, labels=etiquetas)

    print("\n" + "=" * 65)
    print("RESULTADOS")
    print("=" * 65)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"F1 macro:          {f1_macro:.4f}")
    print(f"F1 weighted:       {f1_weighted:.4f}")

    print("\nReporte por clase:")
    print(classification_report(
        y_test, pred, labels=etiquetas, zero_division=0
    ))

    print("Orden de clases de la matriz:", etiquetas)
    print("Matriz de confusión:")
    print(matriz)

    carpeta = PROJECT_ROOT / "models"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_modelo = carpeta / "modelo_v9c_preventivo.pkl"
    ruta_metricas = carpeta / "metricas_modelo_v9c_preventivo.json"
    ruta_predicciones = carpeta / "predicciones_v9c_preventivo.csv"

    joblib.dump(modelo, ruta_modelo)

    metricas = {
        "version": "V9C_preventivo",
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "entrenamiento_hasta": 2018,
        "prueba_desde": 2019,
        "prueba_hasta": 2020,
        "registros_entrenamiento": int(len(train)),
        "registros_prueba": int(len(test)),
        "variables": [str(c) for c in numericas + categoricas],
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "orden_matriz_confusion": etiquetas,
        "matriz_confusion": matriz.tolist(),
        "reporte_por_clase": classification_report(
            y_test, pred, labels=etiquetas,
            output_dict=True, zero_division=0
        ),
        "nota": (
            "Predice la gravedad entre siniestros registrados; "
            "no estima por sí solo la probabilidad de ocurrencia."
        ),
    }

    ruta_metricas.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resultado = pd.DataFrame({
        "GRAVEDAD_REAL": y_test.to_numpy(),
        "GRAVEDAD_PREDICHA": pred,
    })
    resultado.to_csv(
        ruta_predicciones, index=False, encoding="utf-8-sig"
    )

    print("\nArchivos generados:")
    print(ruta_modelo)
    print(ruta_metricas)
    print(ruta_predicciones)


if __name__ == "__main__":
    main()
