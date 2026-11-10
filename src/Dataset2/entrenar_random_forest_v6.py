import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_PATH = "data/processed/siniestros_modelo.csv"
MODEL_PATH = "models/modelo_v6.pkl"
METRICS_PATH = "models/metricas_modelo_v6.json"

RANDOM_STATE = 42


print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V6")
print("MODELO CON VARIABLES ESTRUCTURALES DEL ACCIDENTE")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset original: {df.shape[0]:,} registros x {df.shape[1]} columnas")


# ============================================================
# VARIABLES
# ============================================================

# Variables disponibles antes/durante el evento.
# Se excluyen variables que revelan directamente consecuencias
# o causas determinadas posteriormente.

features = [
    # Información básica del accidente
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",

    # Tiempo
    "FRANJA_HORARIA",
    "ANO",
    "MES",
    "DIA",
    "DIA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "HORA_NUM",

    # Estructura del accidente
    "NUM_ACTORES",
    "NUM_CONDUCTORES",
    "NUM_PEATONES",
    "NUM_VEHICULOS",
    "NUM_VEHICULOS_EN_FUGA",
    "NUM_CLASES_VEHICULO",
    "NUM_SERVICIOS",
    "NUM_MODALIDADES",

    # Tipos de vehículos
    "NUM_AUTOMOVIL",
    "NUM_BUS",
    "NUM_BUSETA",
    "NUM_CAMION_FURGON",
    "NUM_CAMIONETA",
    "NUM_CAMPERO",
    "NUM_MICROBUS",
    "NUM_TRACTOCAMION",
    "NUM_VOLQUETA",
    "NUM_MOTOCICLETA",
    "NUM_BICICLETA",
    "NUM_MOTOCARRO",
    "NUM_BICITAXI",
    "NUM_MOTOS_TOTAL",
]


target = "GRAVEDAD"


# ============================================================
# VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIÓN DE VARIABLES")
print("=" * 70)

missing_features = [col for col in features if col not in df.columns]

if missing_features:
    print("\nERROR: faltan variables:")
    for col in missing_features:
        print(f"  - {col}")
    raise ValueError("Hay variables requeridas que no existen en el dataset.")

print(f"\nVariables utilizadas: {len(features)}")

for col in features:
    print(f"  - {col}")


# ============================================================
# DUPLICADOS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICADOS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicados encontrados: {duplicates:,}")

if duplicates > 0:
    df = df.drop_duplicates().copy()
    print(f"Duplicados eliminados: {duplicates:,}")

print(f"Registros finales: {len(df):,}")


# ============================================================
# VARIABLE OBJETIVO
# ============================================================

print("\n" + "=" * 70)
print("VARIABLE OBJETIVO")
print("=" * 70)

print(target)

print(df[target].value_counts())

print("\nPorcentajes:")
print((df[target].value_counts(normalize=True) * 100).round(2))


# ============================================================
# DATOS FALTANTES
# ============================================================

print("\n" + "=" * 70)
print("DATOS FALTANTES")
print("=" * 70)

missing = df[features + [target]].isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No hay valores faltantes.")
else:
    print(missing)


# ============================================================
# X / Y
# ============================================================

X = df[features].copy()
y = df[target].copy()


# ============================================================
# TIPOS DE VARIABLES
# ============================================================

categorical_features = [
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
    "FRANJA_HORARIA"
]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

print("\n" + "=" * 70)
print("TIPOS DE VARIABLES")
print("=" * 70)

print(f"\nCategóricas: {len(categorical_features)}")
for col in categorical_features:
    print(f"  - {col}")

print(f"\nNuméricas: {len(numeric_features)}")
for col in numeric_features:
    print(f"  - {col}")


# ============================================================
# DIVISIÓN TRAIN / TEST
# ============================================================

print("\n" + "=" * 70)
print("DIVISIÓN DE DATOS")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Entrenamiento: {len(X_train):,}")
print(f"Prueba:        {len(X_test):,}")

print("\nDistribución entrenamiento:")
print(y_train.value_counts())

print("\nDistribución prueba:")
print(y_test.value_counts())


# ============================================================
# PREPROCESAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN RANDOM FOREST V6")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("""
- n_estimators: 400
- max_depth: None
- min_samples_leaf: 2
- max_features: sqrt
- class_weight: balanced_subsample
- random_state: 42
- n_jobs: -1
""")


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print("\nEntrenando Random Forest V6...")
print("Esto puede tardar algunos minutos.")

pipeline.fit(X_train, y_train)

print("\nEntrenamiento completado.")


# ============================================================
# PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_pred = pipeline.predict(X_test)


# ============================================================
# MÉTRICAS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("RESULTADOS RANDOM FOREST V6")
print("=" * 70)

print(f"\nAccuracy:           {accuracy:.4f}")
print(f"Accuracy:           {accuracy * 100:.2f}%")

print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
print(f"Macro Precision:    {macro_precision:.4f}")
print(f"Macro Recall:       {macro_recall:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN")
print("=" * 70)

classes = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

print("\nClases:")
print(classes)

print("\nMatriz:")
print(cm)


# ============================================================
# ANÁLISIS POR CLASE
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS POR CLASE")
print("=" * 70)

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

for clase in classes:

    print(f"\n{clase}")

    print(
        f"  Precision: {report_dict[clase]['precision']:.4f}"
    )

    print(
        f"  Recall:    {report_dict[clase]['recall']:.4f}"
    )

    print(
        f"  F1-score:  {report_dict[clase]['f1-score']:.4f}"
    )

    print(
        f"  Support:   {int(report_dict[clase]['support']):,}"
    )


# ============================================================
# CLASE CRÍTICA
# ============================================================

critical_class = "Con Muertos"

critical_precision = report_dict[critical_class]["precision"]
critical_recall = report_dict[critical_class]["recall"]
critical_f1 = report_dict[critical_class]["f1-score"]
critical_support = int(report_dict[critical_class]["support"])

print("\n" + "=" * 70)
print("ANÁLISIS DE LA CLASE CRÍTICA")
print("=" * 70)

print(f"\nPrecision CON MUERTOS: {critical_precision:.4f}")
print(f"Recall CON MUERTOS:    {critical_recall:.4f}")
print(f"F1 CON MUERTOS:       {critical_f1:.4f}")
print(f"Casos CON MUERTOS:    {critical_support:,}")


# ============================================================
# GUARDAR MODELO
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO MODELO")
print("=" * 70)

os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModelo guardado en:")
print(MODEL_PATH)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metrics = {
    "modelo": "Random Forest V6",
    "registros_utilizados": int(len(df)),
    "variables_predictoras": len(features),
    "variables": features,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "con_muertos": {
        "precision": float(critical_precision),
        "recall": float(critical_recall),
        "f1": float(critical_f1),
        "support": critical_support
    },
    "confusion_matrix": cm.tolist()
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nMétricas guardadas en:")
print(METRICS_PATH)


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL V6")
print("=" * 70)

print(f"""
Registros utilizados: {len(df):,}
Variables predictoras: {len(features)}

Accuracy: {accuracy:.4f}
Balanced Accuracy: {balanced_accuracy:.4f}
Macro F1: {macro_f1:.4f}
F1 Con Muertos: {critical_f1:.4f}
Recall Con Muertos: {critical_recall:.4f}

Modelo:
{MODEL_PATH}

Métricas:
{METRICS_PATH}
""")

print("=" * 70)
print("PROCESO V6 FINALIZADO")
print("=" * 70)