
"""
Comparación controlada:
A = variables temporales + localidad (base V9C).
B = mismas variables + DISENO_LUGAR.

Entrenamiento: hasta 2018.
Prueba: 2019-2020.
Mismos registros, parámetros y métricas para ambos modelos.

No modifica los archivos de V9B, V9C ni V9D.
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
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "siniestros_modelo.csv"
OUTPUT_DIR = ROOT / "models"

SEMILLA = 42
ARBOLES = 150
MIN_MUESTRAS_HOJA = 5


def buscar_columna(df, opciones):
    columnas = {str(c).strip().upper(): c for c in df.columns}
    for opcion in opciones:
        if opcion.upper() in columnas:
            return columnas[opcion.upper()]
    return None


def crear_modelo(numericas, categoricas):
    transformadores = [
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas,
        )
    ]

    if categoricas:
        transformadores.append(
            (
                "categoricas",
                Pipeline([
                    (
                        "imputador",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categoricas,
            )
        )

    preprocesamiento = ColumnTransformer(transformadores)

    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        (
            "clasificador",
            RandomForestClassifier(
                n_estimators=ARBOLES,
                max_features="sqrt",
                min_samples_leaf=MIN_MUESTRAS_HOJA,
                class_weight="balanced_subsample",
                random_state=SEMILLA,
                n_jobs=-1,
                verbose=0,
            ),
        ),
    ])


def metricas_modelo(y_real, pred, etiquetas):
    reporte = classification_report(
        y_real,
        pred,
        labels=etiquetas,
        output_dict=True,
        zero_division=0,
    )

    clase_muertos = next(
        (x for x in etiquetas if "MUERTOS" in x.upper()),
        None,
    )

    if clase_muertos is None:
        clase_muertos = next(
            (x for x in etiquetas if "MUERTO" in x.upper()),
            None,
        )

    if clase_muertos is not None:
        precision_muertos = precision_score(
            y_real,
            pred,
            labels=[clase_muertos],
            average="micro",
            zero_division=0,
        )
        recall_muertos = recall_score(
            y_real,
            pred,
            labels=[clase_muertos],
            average="micro",
            zero_division=0,
        )
        f1_muertos = f1_score(
            y_real,
            pred,
            labels=[clase_muertos],
            average="micro",
            zero_division=0,
        )
    else:
        precision_muertos = 0.0
        recall_muertos = 0.0
        f1_muertos = 0.0

    return {
        "accuracy": float(accuracy_score(y_real, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_real, pred)
        ),
        "f1_macro": float(
            f1_score(
                y_real, pred, average="macro", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_real, pred, average="weighted", zero_division=0
            )
        ),
        "precision_muertos": float(precision_muertos),
        "recall_muertos": float(recall_muertos),
        "f1_muertos": float(f1_muertos),
        "reporte_por_clase": reporte,
        "matriz_confusion": confusion_matrix(
            y_real, pred, labels=etiquetas
        ).tolist(),
        "orden_matriz_confusion": etiquetas,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    objetivo = buscar_columna(df, ["GRAVEDAD"])
    anio = buscar_columna(df, ["ANO", "ANIO", "AÑO"])
    localidad = buscar_columna(
        df, ["CODIGO_LOCALIDAD", "LOCALIDAD", "COD_LOCALIDAD"]
    )
    diseno = buscar_columna(df, ["DISENO_LUGAR", "DISEÑO_LUGAR"])

    mapa = {
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
    for nombre, columna in {
        "GRAVEDAD": objetivo,
        "ANO": anio,
        "CODIGO_LOCALIDAD": localidad,
        "DISENO_LUGAR": diseno,
        **mapa,
    }.items():
        if columna is None:
            faltantes.append(nombre)

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    numericas = [
        anio,
        mapa["MES"],
        mapa["DIA"],
        mapa["DIA_SEMANA"],
        mapa["ES_FIN_DE_SEMANA"],
        mapa["HORA_NUM"],
    ]

    categoricas_base = [localidad]
    categoricas_diseno = [localidad, diseno]

    # Usar las mismas filas en ambos experimentos.
    columnas_necesarias = [
        objetivo, anio, localidad, diseno, *numericas
    ]
    columnas_necesarias = list(dict.fromkeys(columnas_necesarias))

    df = df.copy()

    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[objetivo] = df[objetivo].astype("string").str.strip()
    df = df.dropna(subset=[anio, objetivo]).copy()
    df = df[df[objetivo] != ""].copy()

    # Divisiones temporales idénticas.
    train = df[df[anio] <= 2018].copy()
    test = df[(df[anio] >= 2019) & (df[anio] <= 2020)].copy()

    if train.empty or test.empty:
        raise ValueError(
            "No hay registros suficientes para entrenamiento "
            "hasta 2018 y prueba 2019-2020."
        )

    # Imputar categorías dentro del pipeline.
    for col in [localidad, diseno]:
        train[col] = train[col].astype("string")
        test[col] = test[col].astype("string")

    y_train = train[objetivo].astype(str)
    y_test = test[objetivo].astype(str)

    X_train_base = train[numericas + categoricas_base].copy()
    X_test_base = test[numericas + categoricas_base].copy()

    X_train_diseno = train[numericas + categoricas_diseno].copy()
    X_test_diseno = test[numericas + categoricas_diseno].copy()

    etiquetas = sorted(set(y_train.unique()) | set(y_test.unique()))

    print("=" * 70)
    print("COMPARACIÓN CONTROLADA: BASE V9C VS. + DISENO_LUGAR")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Registros entrenamiento: {len(train):,}")
    print(f"Registros prueba: {len(test):,}")
    print(f"Árboles por modelo: {ARBOLES}")
    print("División temporal: entrenamiento <= 2018; prueba 2019-2020")

    resultados = {}

    experimentos = [
        (
            "A_base",
            X_train_base,
            X_test_base,
            categoricas_base,
        ),
        (
            "B_con_diseno",
            X_train_diseno,
            X_test_diseno,
            categoricas_diseno,
        ),
    ]

    for nombre, X_train, X_test, categoricas in experimentos:
        print(f"\nEntrenando {nombre}...")

        modelo = crear_modelo(numericas, categoricas)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        resultados[nombre] = metricas_modelo(
            y_test, pred, etiquetas
        )

        print(f"\nMétricas de {nombre}:")
        for metrica in [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_muertos",
            "recall_muertos",
            "f1_muertos",
        ]:
            print(f"{metrica}: {resultados[nombre][metrica]:.4f}")

        print("\nReporte por clase:")
        print(classification_report(
            y_test, pred, labels=etiquetas, zero_division=0
        ))

        if nombre == "B_con_diseno":
            ruta_modelo = OUTPUT_DIR / "modelo_comparacion_con_diseno.pkl"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, ruta_modelo)
            print(f"Modelo B guardado en: {ruta_modelo}")

    comparacion = {
        "fecha_ejecucion": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": str(DATA_PATH),
        "entrenamiento_hasta": 2018,
        "prueba_desde": 2019,
        "prueba_hasta": 2020,
        "registros_entrenamiento": int(len(train)),
        "registros_prueba": int(len(test)),
        "parametros": {
            "n_estimators": ARBOLES,
            "min_samples_leaf": MIN_MUESTRAS_HOJA,
            "class_weight": "balanced_subsample",
            "random_state": SEMILLA,
        },
        "variables_base": [str(c) for c in numericas + categoricas_base],
        "variables_con_diseno": [
            str(c) for c in numericas + categoricas_diseno
        ],
        "resultados": resultados,
        "diferencia_B_menos_A": {
            metrica: (
                resultados["B_con_diseno"][metrica]
                - resultados["A_base"][metrica]
            )
            for metrica in [
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_muertos",
                "recall_muertos",
                "f1_muertos",
            ]
        },
        "advertencia": (
            "El valor predictivo no demuestra por sí solo que "
            "DISENO_LUGAR estuviera disponible antes del siniestro. "
            "Verificar definición y origen antes de considerar "
            "el modelo plenamente preventivo."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ruta_json = OUTPUT_DIR / "comparacion_diseno_modelos.json"
    ruta_csv = OUTPUT_DIR / "comparacion_diseno_modelos.csv"

    ruta_json.write_text(
        json.dumps(comparacion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tabla = pd.DataFrame([
        {"modelo": nombre, **{
            metrica: valores[metrica]
            for metrica in [
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_muertos",
                "recall_muertos",
                "f1_muertos",
            ]
        }}
        for nombre, valores in resultados.items()
    ])

    tabla.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(tabla.to_string(index=False))

    print("\nDiferencia B - A:")
    for metrica, diferencia in comparacion["diferencia_B_menos_A"].items():
        print(f"{metrica}: {diferencia:+.4f}")

    print("\nArchivos generados:")
    print(ruta_json)
    print(ruta_csv)


if __name__ == "__main__":
    main()
