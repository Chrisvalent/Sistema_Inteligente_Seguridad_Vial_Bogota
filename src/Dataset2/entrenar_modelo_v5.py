# ============================================================
# ENTRENAMIENTO MODELO V5
# DATASET 2 - SEGURIDAD VIAL BOGOTÁ
#
# Objetivo:
# Evaluar la capacidad predictiva usando únicamente
# variables de contexto del siniestro, evitando variables
# derivadas de las consecuencias del accidente.
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATASET = "data/processed/siniestros_modelo.csv"

RUTA_MODELO = "models/modelo_v5.pkl"
RUTA_METRICAS = "models/metricas_modelo_v5.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# VARIABLES PERMITIDAS
# ============================================================

VARIABLES_CATEGORICAS = [
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
    "FRANJA_HORARIA"
]

VARIABLES_NUMERICAS = [
    "ANO",
    "MES",
    "DIA",
    "DIA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "HORA_NUM"
]

VARIABLES_PERMITIDAS = (
    VARIABLES_CATEGORICAS +
    VARIABLES_NUMERICAS
)

TARGET = "GRAVEDAD"


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V5")
print("MODELO SIN VARIABLES DE CONSECUENCIAS")
print("=" * 70)


os.makedirs("models", exist_ok=True)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATASET)

print(
    f"Dataset original: "
    f"{len(df):,} registros x {len(df.columns)} columnas"
)


# ============================================================
# VALIDAR COLUMNAS
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIÓN DE VARIABLES")
print("=" * 70)

columnas_faltantes = [
    col
    for col in VARIABLES_PERMITIDAS + [TARGET]
    if col not in df.columns
]

if columnas_faltantes:

    print("\nERROR: faltan las siguientes columnas:")

    for col in columnas_faltantes:
        print(f"  - {col}")

    raise ValueError(
        "El dataset no contiene todas las variables necesarias."
    )


print("\nVariables utilizadas:")

for col in VARIABLES_PERMITIDAS:
    print(f"  - {col}")


# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICADOS")
print("=" * 70)

duplicados = df.duplicated().sum()

print(f"Duplicados encontrados: {duplicados}")

if duplicados > 0:

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Duplicados eliminados: {duplicados}"
    )

print(
    f"Registros finales: {len(df):,}"
)


# ============================================================
# VARIABLE OBJETIVO
# ============================================================

y = df[TARGET].copy()

X = df[VARIABLES_PERMITIDAS].copy()


print("\n" + "=" * 70)
print("VARIABLE OBJETIVO")
print("=" * 70)

print(y.value_counts())

print("\nPorcentajes:")

print(
    (y.value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# VALIDACIÓN DE DATOS FALTANTES
# ============================================================

print("\n" + "=" * 70)
print("DATOS FALTANTES")
print("=" * 70)

faltantes = X.isna().sum()

faltantes = faltantes[
    faltantes > 0
].sort_values(ascending=False)

if len(faltantes) == 0:

    print("No hay valores faltantes.")

else:

    print(faltantes)


# ============================================================
# DIVISIÓN TRAIN / TEST
# ============================================================

print("\n" + "=" * 70)
print("DIVISIÓN DE DATOS")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Entrenamiento: {len(X_train):,}"
)

print(
    f"Prueba:        {len(X_test):,}"
)


print("\nDistribución entrenamiento:")

print(
    y_train.value_counts()
)


print("\nDistribución prueba:")

print(
    y_test.value_counts()
)


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

            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )
                ]
            ),

            VARIABLES_NUMERICAS
        ),

        (
            "categoricas",

            Pipeline(
                steps=[

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True
                        )
                    )

                ]
            ),

            VARIABLES_CATEGORICAS
        )

    ],

    remainder="drop"
)


# ============================================================
# MODELO V5
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN RANDOM FOREST V5")
print("=" * 70)

print("""
- n_estimators: 400
- max_depth: None
- min_samples_leaf: 2
- max_features: sqrt
- class_weight: balanced_subsample
- random_state: 42
- n_jobs: -1
""")


modelo = Pipeline(

    steps=[

        (
            "preprocesamiento",
            transformador
        ),

        (
            "modelo",

            RandomForestClassifier(

                n_estimators=400,

                max_depth=None,

                min_samples_leaf=2,

                max_features="sqrt",

                class_weight="balanced_subsample",

                random_state=RANDOM_STATE,

                n_jobs=-1

            )
        )

    ]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print(
    "\nEntrenando Random Forest V5..."
)

print(
    "Esto puede tardar algunos minutos."
)

modelo.fit(
    X_train,
    y_train
)

print(
    "\nEntrenamiento completado."
)


# ============================================================
# PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

predicciones = modelo.predict(
    X_test
)


# ============================================================
# MÉTRICAS GLOBALES
# ============================================================

accuracy = accuracy_score(
    y_test,
    predicciones
)

precision_macro = precision_score(
    y_test,
    predicciones,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    predicciones,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    predicciones,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    predicciones,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS RANDOM FOREST V5")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"\nMacro Precision: {precision_macro:.4f}"
)

print(
    f"Macro Recall:    {recall_macro:.4f}"
)

print(
    f"Macro F1:        {f1_macro:.4f}"
)

print(
    f"Weighted F1:     {f1_weighted:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

reporte_texto = classification_report(
    y_test,
    predicciones,
    zero_division=0
)

print(
    reporte_texto
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN")
print("=" * 70)

matriz = confusion_matrix(
    y_test,
    predicciones
)

clases = sorted(
    y.unique()
)

print("\nClases:")

print(clases)

print("\nMatriz:")

print(matriz)


# ============================================================
# MÉTRICAS POR CLASE
# ============================================================

reporte = classification_report(
    y_test,
    predicciones,
    output_dict=True,
    zero_division=0
)


print("\n" + "=" * 70)
print("ANÁLISIS POR CLASE")
print("=" * 70)


for clase in clases:

    if clase not in reporte:
        continue

    print(f"\n{clase}")

    print(
        f"  Precision: "
        f"{reporte[clase]['precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{reporte[clase]['recall']:.4f}"
    )

    print(
        f"  F1-score:  "
        f"{reporte[clase]['f1-score']:.4f}"
    )

    print(
        f"  Support:   "
        f"{int(reporte[clase]['support'])}"
    )


# ============================================================
# ANÁLISIS ESPECÍFICO DE CON MUERTOS
# ============================================================

CLASE_CRITICA = "Con Muertos"

print("\n" + "=" * 70)
print("ANÁLISIS DE LA CLASE CRÍTICA")
print("=" * 70)


if CLASE_CRITICA in reporte:

    precision_muertos = (
        reporte[CLASE_CRITICA]["precision"]
    )

    recall_muertos = (
        reporte[CLASE_CRITICA]["recall"]
    )

    f1_muertos = (
        reporte[CLASE_CRITICA]["f1-score"]
    )

    support_muertos = (
        reporte[CLASE_CRITICA]["support"]
    )

    print(
        f"\nPrecision CON MUERTOS: "
        f"{precision_muertos:.4f}"
    )

    print(
        f"Recall CON MUERTOS:    "
        f"{recall_muertos:.4f}"
    )

    print(
        f"F1 CON MUERTOS:       "
        f"{f1_muertos:.4f}"
    )

    print(
        f"Casos CON MUERTOS:    "
        f"{int(support_muertos)}"
    )

else:

    precision_muertos = 0
    recall_muertos = 0
    f1_muertos = 0
    support_muertos = 0

    print(
        "No se encontró la clase CON MUERTOS."
    )


# ============================================================
# GUARDAR MODELO
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO MODELO")
print("=" * 70)


joblib.dump(
    modelo,
    RUTA_MODELO
)

print(
    f"\nModelo guardado en:"
)

print(
    RUTA_MODELO
)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = {

    "modelo": "Random Forest V5",

    "descripcion":
        "Modelo sin variables derivadas de consecuencias "
        "de actores, vehículos o hipótesis.",

    "dataset":
        "data/processed/siniestros_modelo.csv",

    "registros_originales":
        int(len(pd.read_csv(RUTA_DATASET))),

    "registros_finales":
        int(len(df)),

    "duplicados_eliminados":
        int(duplicados),

    "variables_utilizadas":
        VARIABLES_PERMITIDAS,

    "variables_excluidas":

        [
            "GRAVEDAD",
            "CODIGO_ACCIDENTE",
            "DIRECCION",
            "NUM_ILESOS",
            "NUM_HERIDOS_ACTOR",
            "NUM_FALLECIDOS_ACTOR"
        ],

    "accuracy":
        float(accuracy),

    "macro_precision":
        float(precision_macro),

    "macro_recall":
        float(recall_macro),

    "macro_f1":
        float(f1_macro),

    "weighted_f1":
        float(f1_weighted),

    "con_muertos":

        {
            "precision":
                float(precision_muertos),

            "recall":
                float(recall_muertos),

            "f1":
                float(f1_muertos),

            "support":
                int(support_muertos)
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


print(
    "\nMétricas guardadas en:"
)

print(
    RUTA_METRICAS
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL V5")
print("=" * 70)

print(
    f"\nRegistros utilizados: "
    f"{len(df):,}"
)

print(
    f"Variables predictoras: "
    f"{len(VARIABLES_PERMITIDAS)}"
)

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Macro F1: "
    f"{f1_macro:.4f}"
)

print(
    f"F1 Con Muertos: "
    f"{f1_muertos:.4f}"
)

print(
    f"Recall Con Muertos: "
    f"{recall_muertos:.4f}"
)

print("\n" + "=" * 70)
print("PROCESO V5 FINALIZADO")
print("=" * 70)