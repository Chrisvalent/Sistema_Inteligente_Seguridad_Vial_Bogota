
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

# ============================================================
# V9B - EVALUACION TEMPORAL ESTRICTA
# Entrenamiento: 2015-2018
# Prueba:        2019-2020
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "siniestros_modelo.csv"
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "modelo_v9b.pkl"
METRICS_PATH = MODELS_DIR / "metricas_modelo_v9b.json"
PREDICTIONS_PATH = MODELS_DIR / "predicciones_v9b.csv"

TARGET = "GRAVEDAD"
THRESHOLD_MUERTOS = 0.30
RANDOM_STATE = 42

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
    "NUM_BICITAXI", "NUM_MOTOS_TOTAL"
]

CATEGORICAS = [
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
    "FRANJA_HORARIA",
]

NUMERICAS = [
    columna for columna in VARIABLES
    if columna not in CATEGORICAS
]

CLASES = ["Con Heridos", "Con Muertos", "Solo Daños"]


def main():
    print("=" * 65)
    print("V9B - EVALUACION TEMPORAL ESTRICTA")
    print("=" * 65)

    # 1. Cargar datos
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"\nDatos cargados: {df.shape[0]:,} filas")
    print(f"Columnas originales: {df.shape[1]}")

    # 2. Verificar columnas necesarias
    requeridas = VARIABLES + [TARGET]
    faltantes = [c for c in requeridas if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias en el CSV: {faltantes}"
        )

    # 3. Limpiar target y año
    df = df.dropna(subset=[TARGET, "ANO"]).copy()

    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")
    df = df.dropna(subset=["ANO"]).copy()
    df["ANO"] = df["ANO"].astype(int)

    # Mantener solo los años de estudio
    df = df[df["ANO"].between(2015, 2020)].copy()

    # Eliminar duplicados exactos
    filas_antes = len(df)
    df = df.drop_duplicates().copy()
    print(f"Duplicados exactos eliminados: {filas_antes - len(df):,}")

    # Verificar las clases esperadas
    clases_observadas = set(df[TARGET].unique())
    clases_inesperadas = clases_observadas - set(CLASES)

    if clases_inesperadas:
        raise ValueError(
            f"Hay etiquetas inesperadas en {TARGET}: "
            f"{clases_inesperadas}"
        )

    # 4. Separar por año, sin solapamiento temporal
    train = df[df["ANO"] <= 2018].copy()
    test = df[df["ANO"] >= 2019].copy()

    if train.empty or test.empty:
        raise ValueError(
            "El conjunto de entrenamiento o prueba está vacío."
        )

    print("\nDistribución temporal:")
    print("Entrenamiento:")
    print(train["ANO"].value_counts().sort_index().to_string())

    print("\nPrueba:")
    print(test["ANO"].value_counts().sort_index().to_string())

    print("\nDistribución de GRAVEDAD - entrenamiento:")
    print(train[TARGET].value_counts().to_string())

    print("\nDistribución de GRAVEDAD - prueba:")
    print(test[TARGET].value_counts().to_string())

    years_train = set(train["ANO"].unique())
    years_test = set(test["ANO"].unique())

    print(f"\nAños de entrenamiento: {sorted(years_train)}")
    print(f"Años de prueba: {sorted(years_test)}")

    if years_train.intersection(years_test):
        raise RuntimeError(
            "Error: hay años compartidos entre entrenamiento y prueba."
        )

    print("\nVerificación temporal: OK, no hay años compartidos.")

    # 5. Preparar X e y
    X_train = train[VARIABLES].copy()
    y_train = train[TARGET].copy()

    X_test = test[VARIABLES].copy()
    y_test = test[TARGET].copy()

    # 6. Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERICAS,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAS,
            ),
        ],
        remainder="drop",
    )

    # 7. Modelo
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={
            "Con Heridos": 1.0,
            "Con Muertos": 5.0,
            "Solo Daños": 1.0,
        },
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    # 8. Entrenar solo con 2015-2018
    print("\nEntrenando el modelo...")
    pipeline.fit(X_train, y_train)
    print("Entrenamiento finalizado.")

    # 9. Predecir sobre 2019-2020
    # El umbral de 0.30 se conserva de V8B.
    probabilidades = pipeline.predict_proba(X_test)
    clases_modelo = list(pipeline.named_steps["model"].classes_)

    idx_muertos = clases_modelo.index("Con Muertos")
    prob_muertos = probabilidades[:, idx_muertos]

    # Predicción normal por máxima probabilidad
    indices_maximos = probabilidades.argmax(axis=1)
    y_pred = [
        clases_modelo[i] for i in indices_maximos
    ]

    # Ajustar la decisión para priorizar la detección de muertos
    for i, prob in enumerate(prob_muertos):
        if prob >= THRESHOLD_MUERTOS:
            y_pred[i] = "Con Muertos"

    # 10. Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    precision_muertos = precision_score(
        y_test,
        y_pred,
        labels=["Con Muertos"],
        average="micro",
        zero_division=0,
    )

    recall_muertos = recall_score(
        y_test,
        y_pred,
        labels=["Con Muertos"],
        average="micro",
        zero_division=0,
    )

    f1_muertos = f1_score(
        y_test,
        y_pred,
        labels=["Con Muertos"],
        average="micro",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASES)

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASES,
        output_dict=True,
        zero_division=0,
    )

    # 11. Mostrar resultados
    print("\n" + "=" * 65)
    print("RESULTADOS V9B - PRUEBA TEMPORAL 2019-2020")
    print("=" * 65)

    print(f"Registros de entrenamiento: {len(X_train):,}")
    print(f"Registros de prueba:        {len(X_test):,}")
    print(f"Umbral para Con Muertos:    {THRESHOLD_MUERTOS:.2f}")

    print(f"\nAccuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"F1 macro:           {macro_f1:.4f}")

    print("\nMétricas de la clase Con Muertos:")
    print(f"Precision: {precision_muertos:.4f}")
    print(f"Recall:    {recall_muertos:.4f}")
    print(f"F1:        {f1_muertos:.4f}")

    print("\nReporte por clase:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASES,
            zero_division=0,
        )
    )

    print("Matriz de confusión")
    print(f"Orden de clases: {CLASES}")
    print(cm)

    # 12. Guardar modelo y metadatos
    model_artifact = {
        "pipeline": pipeline,
        "variables": VARIABLES,
        "target": TARGET,
        "threshold_con_muertos": THRESHOLD_MUERTOS,
        "clases": clases_modelo,
        "version": "V9B",
        "train_years": sorted(years_train),
        "test_years": sorted(years_test),
    }

    joblib.dump(model_artifact, MODEL_PATH)

    metrics = {
        "version": "V9B",
        "train_years": sorted(int(y) for y in years_train),
        "test_years": sorted(int(y) for y in years_test),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "threshold_con_muertos": THRESHOLD_MUERTOS,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "f1_macro": float(macro_f1),
        "con_muertos": {
            "precision": float(precision_muertos),
            "recall": float(recall_muertos),
            "f1": float(f1_muertos),
            "casos_reales": int((y_test == "Con Muertos").sum()),
            "casos_detectados": int(
                ((y_test == "Con Muertos") &
                 (pd.Series(y_pred, index=y_test.index) == "Con Muertos"))
                .sum()
            ),
            "predicciones": int(
                (pd.Series(y_pred, index=y_test.index) == "Con Muertos")
                .sum()
            ),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASES,
        "classification_report": report,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # Guardar predicciones para análisis posterior
    predictions = pd.DataFrame({
        "ANO": test["ANO"].values,
        "GRAVEDAD_REAL": y_test.values,
        "GRAVEDAD_PREDICHA": y_pred,
        "PROBABILIDAD_CON_MUERTOS": prob_muertos,
    })

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nArchivos guardados:")
    print(f"Modelo:       {MODEL_PATH}")
    print(f"Métricas:     {METRICS_PATH}")
    print(f"Predicciones:{PREDICTIONS_PATH}")

    print("\nV9B finalizado correctamente.")


if __name__ == "__main__":
    main()
