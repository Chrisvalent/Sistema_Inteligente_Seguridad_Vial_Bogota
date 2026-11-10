import pandas as pd
import numpy as np
import json
import joblib

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
# 1. CARGAR DATASET
# ============================================================

RUTA_DATASET = "data/processed/siniestros_modelo.csv"

df = pd.read_csv(RUTA_DATASET)

print("=" * 70)
print("MODELO V7 - RANDOM FOREST CON MAYOR PESO PARA CON MUERTOS")
print("=" * 70)

print(f"Dataset original: {df.shape}")


# ============================================================
# 2. ELIMINAR DUPLICADOS
# ============================================================

duplicados = df.duplicated().sum()
df = df.drop_duplicates().copy()

print(f"Duplicados eliminados: {duplicados}")
print(f"Dataset final: {df.shape}")


# ============================================================
# 3. VARIABLE OBJETIVO
# ============================================================

TARGET = "GRAVEDAD"

X = df.drop(columns=[TARGET])
y = df[TARGET]


print("\nDistribución de la variable objetivo:")
print(y.value_counts())
print("\nDistribución porcentual:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 4. VARIABLES DEL MODELO V6
# ============================================================

variables_categoricas = [
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
    "FRANJA_HORARIA"
]

variables_numericas = [
    "ANO",
    "MES",
    "DIA",
    "DIA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "HORA_NUM",

    "NUM_ACTORES",
    "NUM_CONDUCTORES",
    "NUM_PEATONES",
    "NUM_VEHICULOS",
    "NUM_VEHICULOS_EN_FUGA",
    "NUM_CLASES_VEHICULO",
    "NUM_SERVICIOS",
    "NUM_MODALIDADES",

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
    "NUM_MOTOS_TOTAL"
]

VARIABLES = variables_categoricas + variables_numericas

X = X[VARIABLES].copy()


# ============================================================
# 5. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)


# ============================================================
# 6. PREPROCESAMIENTO
# ============================================================

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            variables_categoricas
        ),
        (
            "numericas",
            "passthrough",
            variables_numericas
        )
    ]
)


# ============================================================
# 7. PESOS DE CLASE
# ============================================================

# V6 utilizaba:
# class_weight="balanced_subsample"
#
# En V7 damos mayor importancia a CON MUERTOS.

pesos_clase = {
    "Con Heridos": 1.0,
    "Con Muertos": 5.0,
    "Solo Daños": 1.0
}

print("\nPesos utilizados:")
print(pesos_clase)


# ============================================================
# 8. RANDOM FOREST V7
# ============================================================

modelo = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=pesos_clase,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("modelo", modelo)
    ]
)


# ============================================================
# 9. ENTRENAMIENTO
# ============================================================

print("\nEntrenando modelo V7...")

pipeline.fit(X_train, y_train)

print("Entrenamiento terminado.")


# ============================================================
# 10. PREDICCIONES
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 11. MÉTRICAS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

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
print("RESULTADOS V7")
print("=" * 70)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")


# ============================================================
# 12. REPORTE POR CLASE
# ============================================================

print("\nREPORTE POR CLASE:")
print()

reporte = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(reporte)


# ============================================================
# 13. MATRIZ DE CONFUSIÓN
# ============================================================

clases = [
    "Con Heridos",
    "Con Muertos",
    "Solo Daños"
]

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=clases
)

print("MATRIZ DE CONFUSIÓN:")
print()

print(
    pd.DataFrame(
        matriz,
        index=clases,
        columns=clases
    )
)


# ============================================================
# 14. GUARDAR MODELO
# ============================================================

RUTA_MODELO = "models/modelo_v7.pkl"

joblib.dump(
    pipeline,
    RUTA_MODELO
)

print(f"\nModelo guardado en: {RUTA_MODELO}")


# ============================================================
# 15. GUARDAR MÉTRICAS
# ============================================================

reporte_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

metricas = {
    "modelo": "Random Forest V7",
    "n_registros": int(len(df)),
    "n_variables": int(len(VARIABLES)),
    "n_variables_categoricas": int(len(variables_categoricas)),
    "n_variables_numericas": int(len(variables_numericas)),

    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),

    "pesos_clase": pesos_clase,
    "reporte_clasificacion": reporte_dict,
    "matriz_confusion": matriz.tolist()
}

RUTA_METRICAS = "models/metricas_modelo_v7.json"

with open(
    RUTA_METRICAS,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        metricas,
        archivo,
        indent=4,
        ensure_ascii=False
    )

print(f"Métricas guardadas en: {RUTA_METRICAS}")

print("\nV7 terminado correctamente.")