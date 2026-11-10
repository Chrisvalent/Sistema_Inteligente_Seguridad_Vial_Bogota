# ============================================================
# ENTRENAMIENTO MODELO V4 - DATASET 2
# Seguridad Vial Bogotá
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATASET = "data/processed/siniestros_modelo.csv"
RUTA_MODELO = "models/modelo_v4.pkl"
RUTA_METRICAS = "models/metricas_modelo_v4.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO MODELO V4 - DATASET 2")
print("=" * 70)

# Crear carpeta de modelos si no existe
os.makedirs("models", exist_ok=True)

# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATASET)

print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

duplicados = df.duplicated().sum()

print(f"\nDuplicados completos encontrados: {duplicados}")

if duplicados > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados eliminados: {duplicados}")
    print(f"Dataset final: {len(df):,} filas")

# ============================================================
# VARIABLE OBJETIVO
# ============================================================

TARGET = "GRAVEDAD"

if TARGET not in df.columns:
    raise ValueError(f"No existe la columna objetivo '{TARGET}'")

y = df[TARGET].copy()
X = df.drop(columns=[TARGET]).copy()

print("\n" + "=" * 70)
print("DISTRIBUCIÓN DEL OBJETIVO")
print("=" * 70)

print(y.value_counts())

print("\nPorcentajes:")
print((y.value_counts(normalize=True) * 100).round(2))

# ============================================================
# IDENTIFICAR VARIABLES CATEGÓRICAS Y NUMÉRICAS
# ============================================================

categoricas = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numericas = X.select_dtypes(
    include=[np.number, "bool"]
).columns.tolist()

print("\n" + "=" * 70)
print("TIPOS DE VARIABLES")
print("=" * 70)

print(f"Variables categóricas: {len(categoricas)}")
print(f"Variables numéricas:   {len(numericas)}")
print(f"Total predictores:     {X.shape[1]}")

print("\nCategóricas:")
for col in categoricas:
    print(f"  - {col}")

# ============================================================
# DIVISIÓN ESTRATIFICADA
# ============================================================

print("\n" + "=" * 70)
print("DIVISIÓN TRAIN / TEST")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Entrenamiento: {len(X_train):,} registros")
print(f"Prueba:        {len(X_test):,} registros")

print("\nDistribución TRAIN:")
print(y_train.value_counts())

print("\nDistribución TEST:")
print(y_test.value_counts())

# ============================================================
# PREPROCESAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

transformador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numericas
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True
                    )
                )
            ]),
            categoricas
        )
    ],
    remainder="drop"
)

# ============================================================
# MODELO 1: RANDOM FOREST BALANCEADO
# ============================================================

print("\n" + "=" * 70)
print("MODELO 1 - RANDOM FOREST BALANCEADO")
print("=" * 70)

modelo_balanceado = Pipeline([
    ("preprocesamiento", transformador),
    (
        "modelo",
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    )
])

print("Entrenando modelo balanceado...")
modelo_balanceado.fit(X_train, y_train)

print("Entrenamiento terminado.")

pred_balanceado = modelo_balanceado.predict(X_test)

# ============================================================
# EVALUACIÓN MODELO 1
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS - MODELO 1")
print("=" * 70)

accuracy_1 = accuracy_score(y_test, pred_balanceado)
macro_f1_1 = f1_score(
    y_test,
    pred_balanceado,
    average="macro"
)

macro_recall_1 = recall_score(
    y_test,
    pred_balanceado,
    average="macro"
)

macro_precision_1 = precision_score(
    y_test,
    pred_balanceado,
    average="macro",
    zero_division=0
)

print(f"\nAccuracy:         {accuracy_1:.4f}")
print(f"Macro Precision:  {macro_precision_1:.4f}")
print(f"Macro Recall:     {macro_recall_1:.4f}")
print(f"Macro F1:         {macro_f1_1:.4f}")

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        pred_balanceado,
        zero_division=0
    )
)

print("Matriz de confusión:")
print(confusion_matrix(y_test, pred_balanceado))

# ============================================================
# MODELO 2: RANDOM FOREST CON MAYOR PESO A CON MUERTOS
# ============================================================

print("\n" + "=" * 70)
print("MODELO 2 - RANDOM FOREST ENFOCADO EN CON MUERTOS")
print("=" * 70)

# Primero calculamos pesos a partir de las clases
conteos = y_train.value_counts()

peso_muertos = max(
    1.0,
    conteos.max() / conteos.get("Con Muertos", 1)
)

# Reducimos el peso extremo para evitar que el modelo
# clasifique excesivamente como "Con Muertos".
peso_muertos = min(peso_muertos, 15.0)

pesos = {
    "Solo Daños": 1.0,
    "Con Heridos": 1.5,
    "Con Muertos": peso_muertos
}

print("Pesos utilizados:")
for clase, peso in pesos.items():
    print(f"  {clase}: {peso:.2f}")

modelo_muertos = Pipeline([
    ("preprocesamiento", transformador),
    (
        "modelo",
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=pesos,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    )
])

print("\nEntrenando modelo enfocado en Con Muertos...")
modelo_muertos.fit(X_train, y_train)

print("Entrenamiento terminado.")

pred_muertos = modelo_muertos.predict(X_test)

# ============================================================
# EVALUACIÓN MODELO 2
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS - MODELO 2")
print("=" * 70)

accuracy_2 = accuracy_score(y_test, pred_muertos)

macro_f1_2 = f1_score(
    y_test,
    pred_muertos,
    average="macro"
)

macro_recall_2 = recall_score(
    y_test,
    pred_muertos,
    average="macro"
)

macro_precision_2 = precision_score(
    y_test,
    pred_muertos,
    average="macro",
    zero_division=0
)

print(f"\nAccuracy:         {accuracy_2:.4f}")
print(f"Macro Precision:  {macro_precision_2:.4f}")
print(f"Macro Recall:     {macro_recall_2:.4f}")
print(f"Macro F1:         {macro_f1_2:.4f}")

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        pred_muertos,
        zero_division=0
    )
)

print("Matriz de confusión:")
print(confusion_matrix(y_test, pred_muertos))

# ============================================================
# RECALL ESPECÍFICO DE CON MUERTOS
# ============================================================

def obtener_metricas_clase(y_real, y_pred, clase):
    reporte = classification_report(
        y_real,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    if clase in reporte:
        return {
            "precision": reporte[clase]["precision"],
            "recall": reporte[clase]["recall"],
            "f1": reporte[clase]["f1-score"],
            "support": reporte[clase]["support"]
        }

    return {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "support": 0
    }


metricas_muertos_1 = obtener_metricas_clase(
    y_test,
    pred_balanceado,
    "Con Muertos"
)

metricas_muertos_2 = obtener_metricas_clase(
    y_test,
    pred_muertos,
    "Con Muertos"
)

print("\n" + "=" * 70)
print("DETECCIÓN DE CON MUERTOS")
print("=" * 70)

print("\nModelo 1:")
print(
    f"Precision: {metricas_muertos_1['precision']:.4f} | "
    f"Recall: {metricas_muertos_1['recall']:.4f} | "
    f"F1: {metricas_muertos_1['f1']:.4f}"
)

print("\nModelo 2:")
print(
    f"Precision: {metricas_muertos_2['precision']:.4f} | "
    f"Recall: {metricas_muertos_2['recall']:.4f} | "
    f"F1: {metricas_muertos_2['f1']:.4f}"
)

# ============================================================
# SELECCIONAR MEJOR MODELO
# ============================================================

# Priorizamos F1 macro.
# En caso de empate, priorizamos F1 de Con Muertos.

if (
    macro_f1_2 > macro_f1_1
    or (
        np.isclose(macro_f1_2, macro_f1_1)
        and metricas_muertos_2["f1"] > metricas_muertos_1["f1"]
    )
):
    mejor_modelo = modelo_muertos
    mejor_nombre = "Modelo 2 - Enfocado en Con Muertos"
    mejor_pred = pred_muertos
    mejor_macro_f1 = macro_f1_2
    mejor_muertos = metricas_muertos_2

else:
    mejor_modelo = modelo_balanceado
    mejor_nombre = "Modelo 1 - Random Forest Balanceado"
    mejor_pred = pred_balanceado
    mejor_macro_f1 = macro_f1_1
    mejor_muertos = metricas_muertos_1

# ============================================================
# GUARDAR MODELO
# ============================================================

print("\n" + "=" * 70)
print("SELECCIÓN DEL MEJOR MODELO")
print("=" * 70)

print(f"\nModelo seleccionado:")
print(mejor_nombre)

print(f"Macro F1: {mejor_macro_f1:.4f}")
print(
    f"F1 Con Muertos: "
    f"{mejor_muertos['f1']:.4f}"
)

joblib.dump(
    mejor_modelo,
    RUTA_MODELO
)

print(f"\nModelo guardado en:")
print(RUTA_MODELO)

# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = {
    "dataset": "siniestros_modelo.csv",
    "registros_originales": int(196152),
    "registros_despues_duplicados": int(len(df)),
    "duplicados_eliminados": int(duplicados),
    "variables_predictoras": int(X.shape[1]),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,

    "modelo_1": {
        "nombre": "Random Forest Balanceado",
        "accuracy": float(accuracy_1),
        "macro_precision": float(macro_precision_1),
        "macro_recall": float(macro_recall_1),
        "macro_f1": float(macro_f1_1),
        "con_muertos": metricas_muertos_1
    },

    "modelo_2": {
        "nombre": "Random Forest Enfocado en Con Muertos",
        "accuracy": float(accuracy_2),
        "macro_precision": float(macro_precision_2),
        "macro_recall": float(macro_recall_2),
        "macro_f1": float(macro_f1_2),
        "con_muertos": metricas_muertos_2
    },

    "mejor_modelo": {
        "nombre": mejor_nombre,
        "macro_f1": float(mejor_macro_f1),
        "con_muertos_f1": float(mejor_muertos["f1"]),
        "con_muertos_recall": float(mejor_muertos["recall"])
    }
}

with open(
    RUTA_METRICAS,
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(
        metricas,
        archivo,
        ensure_ascii=False,
        indent=4
    )

print(f"Métricas guardadas en:")
print(RUTA_METRICAS)

# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL MODELO V4")
print("=" * 70)

print(
    f"\n{'Modelo':45} "
    f"{'Accuracy':>10} "
    f"{'Macro F1':>10} "
    f"{'F1 Muertos':>12}"
)

print("-" * 82)

print(
    f"{'RF Balanceado':45} "
    f"{accuracy_1:>10.4f} "
    f"{macro_f1_1:>10.4f} "
    f"{metricas_muertos_1['f1']:>12.4f}"
)

print(
    f"{'RF Enfocado en Muertos':45} "
    f"{accuracy_2:>10.4f} "
    f"{macro_f1_2:>10.4f} "
    f"{metricas_muertos_2['f1']:>12.4f}"
)

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)