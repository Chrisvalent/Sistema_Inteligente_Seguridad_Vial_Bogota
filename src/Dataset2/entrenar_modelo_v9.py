import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATOS = "data/processed/siniestros_modelo.csv"

RUTA_MODELO = "models/modelo_v9.pkl"
RUTA_METRICAS = "models/metricas_modelo_v9.json"

VARIABLE_OBJETIVO = "GRAVEDAD"

VARIABLES = [
    "CLASE",
    "CHOQUE",
    "OBJETO_FIJO",
    "CODIGO_LOCALIDAD",
    "DISENO_LUGAR",
    "FRANJA_HORARIA",
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

CLASES = [
    "Con Heridos",
    "Con Muertos",
    "Solo Daños"
]

PESOS_CLASES = {
    "Con Heridos": 1.0,
    "Con Muertos": 5.0,
    "Solo Daños": 1.0
}

# Umbral obtenido previamente mediante validación en V8B.
UMBRAL_MUERTOS = 0.30


# ============================================================
# FUNCIÓN DE UMBRAL
# ============================================================

def aplicar_umbral(probabilidades, clases, umbral):

    indice_muertos = list(clases).index("Con Muertos")

    predicciones_base = clases[
        np.argmax(probabilidades, axis=1)
    ]

    predicciones = predicciones_base.copy()

    mascara_muertos = (
        probabilidades[:, indice_muertos] >= umbral
    )

    predicciones[mascara_muertos] = "Con Muertos"

    return predicciones


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V9")
print("VALIDACIÓN TEMPORAL")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATOS)

print(f"Dataset original: {df.shape}")


# ============================================================
# DUPLICADOS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICADOS")
print("=" * 70)

duplicados = df.duplicated().sum()

print(f"Duplicados encontrados: {duplicados}")

df = df.drop_duplicates().copy()

print(f"Registros finales: {len(df):,}")


# ============================================================
# REVISAR AÑOS
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUCIÓN TEMPORAL")
print("=" * 70)

print("\nRegistros por año:")

print(
    df["ANO"]
    .value_counts()
    .sort_index()
)


# ============================================================
# ORDENAR TEMPORALMENTE
# ============================================================

df = df.sort_values(
    by=["ANO", "MES", "DIA", "HORA_NUM"]
).reset_index(drop=True)


# ============================================================
# DIVISIÓN TEMPORAL
# ============================================================

print("\n" + "=" * 70)
print("DIVISIÓN TEMPORAL")
print("=" * 70)

anios = sorted(
    df["ANO"].dropna().unique()
)

print("\nAños disponibles:")

for anio in anios:
    cantidad = (df["ANO"] == anio).sum()
    print(f"{int(anio)}: {cantidad:,}")


# ------------------------------------------------------------
# IMPORTANTE:
#
# Se reserva aproximadamente el último 20% temporal
# como evaluación futura.
#
# El punto de corte se determina por posición temporal.
# ------------------------------------------------------------

indice_corte = int(len(df) * 0.80)

df_train = df.iloc[:indice_corte].copy()
df_test = df.iloc[indice_corte:].copy()


print("\n" + "-" * 70)

print(
    f"Entrenamiento temporal: "
    f"{len(df_train):,}"
)

print(
    f"Prueba temporal:        "
    f"{len(df_test):,}"
)

print(
    f"Porcentaje entrenamiento: "
    f"{len(df_train) / len(df) * 100:.2f}%"
)

print(
    f"Porcentaje prueba:        "
    f"{len(df_test) / len(df) * 100:.2f}%"
)


print("\nPeriodo entrenamiento:")

print(
    f"Desde {int(df_train['ANO'].min())}"
    f" hasta {int(df_train['ANO'].max())}"
)

print("\nPeriodo prueba:")

print(
    f"Desde {int(df_test['ANO'].min())}"
    f" hasta {int(df_test['ANO'].max())}"
)


print("\nDistribución entrenamiento:")

print(
    df_train[VARIABLE_OBJETIVO]
    .value_counts()
)


print("\nDistribución prueba:")

print(
    df_test[VARIABLE_OBJETIVO]
    .value_counts()
)


# ============================================================
# X / y
# ============================================================

X_train = df_train[VARIABLES].copy()
y_train = df_train[VARIABLE_OBJETIVO].copy()

X_test = df_test[VARIABLES].copy()
y_test = df_test[VARIABLE_OBJETIVO].copy()


# ============================================================
# PREPROCESAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

transformador_numerico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

transformador_categorico = Pipeline(
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
                sparse_output=False
            )
        )
    ]
)

preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            transformador_numerico,
            VARIABLES_NUMERICAS
        ),
        (
            "categoricas",
            transformador_categorico,
            VARIABLES_CATEGORICAS
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN RANDOM FOREST V9")
print("=" * 70)

print("- n_estimators: 400")
print("- max_depth: None")
print("- min_samples_leaf: 2")
print("- max_features: sqrt")
print(
    f"- class_weight: {PESOS_CLASES}"
)
print("- random_state: 42")
print("- n_jobs: -1")
print(
    f"- umbral Con Muertos: {UMBRAL_MUERTOS}"
)


modelo_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=PESOS_CLASES,
    random_state=42,
    n_jobs=-1
)


modelo = Pipeline(
    steps=[
        (
            "preprocesamiento",
            preprocesamiento
        ),
        (
            "modelo",
            modelo_rf
        )
    ]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print("\nEntrenando con datos históricos...")

modelo.fit(
    X_train,
    y_train
)

print("Entrenamiento completado.")


# ============================================================
# PREDICCIÓN
# ============================================================

print("\n" + "=" * 70)
print("PREDICCIÓN SOBRE PERIODO FUTURO")
print("=" * 70)

print("\nGenerando probabilidades...")

prob_test = modelo.predict_proba(
    X_test
)

print("Probabilidades generadas.")


pred_test = aplicar_umbral(
    prob_test,
    modelo.classes_,
    UMBRAL_MUERTOS
)


# ============================================================
# MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_test,
    pred_test
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    pred_test
)

macro_f1 = f1_score(
    y_test,
    pred_test,
    average="macro",
    zero_division=0
)

precision_muertos = precision_score(
    y_test,
    pred_test,
    labels=["Con Muertos"],
    average="macro",
    zero_division=0
)

recall_muertos = recall_score(
    y_test,
    pred_test,
    labels=["Con Muertos"],
    average="macro",
    zero_division=0
)

f1_muertos = f1_score(
    y_test,
    pred_test,
    labels=["Con Muertos"],
    average="macro",
    zero_division=0
)

predicciones_muertos = np.sum(
    pred_test == "Con Muertos"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - V9")
print("=" * 70)

print(
    classification_report(
        y_test,
        pred_test,
        labels=CLASES,
        zero_division=0
    )
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN - V9")
print("=" * 70)

matriz = confusion_matrix(
    y_test,
    pred_test,
    labels=CLASES
)

print(
    pd.DataFrame(
        matriz,
        index=CLASES,
        columns=CLASES
    )
)


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL V9")
print("=" * 70)

print(
    f"Registros utilizados:       {len(df):,}"
)

print(
    f"Variables predictoras:      {len(VARIABLES)}"
)

print(
    f"Umbral Con Muertos:         {UMBRAL_MUERTOS:.3f}"
)

print(
    f"\nAccuracy:                   {accuracy:.4f}"
)

print(
    f"Balanced Accuracy:          "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro F1:                   {macro_f1:.4f}"
)

print(
    f"\nPrecision Con Muertos:      "
    f"{precision_muertos:.4f}"
)

print(
    f"Recall Con Muertos:         "
    f"{recall_muertos:.4f}"
)

print(
    f"F1 Con Muertos:             "
    f"{f1_muertos:.4f}"
)

print(
    f"\nPredicciones Con Muertos:   "
    f"{predicciones_muertos}"
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
    "Modelo guardado en:"
)

print(RUTA_MODELO)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = {
    "modelo": "Random Forest V9",
    "tipo_validacion": "temporal",
    "registros": int(len(df)),
    "registros_entrenamiento": int(len(df_train)),
    "registros_prueba": int(len(df_test)),
    "variables_predictoras": int(len(VARIABLES)),
    "umbral_con_muertos": float(
        UMBRAL_MUERTOS
    ),

    "periodo_entrenamiento": {
        "anio_inicio": int(
            df_train["ANO"].min()
        ),
        "anio_fin": int(
            df_train["ANO"].max()
        )
    },

    "periodo_prueba": {
        "anio_inicio": int(
            df_test["ANO"].min()
        ),
        "anio_fin": int(
            df_test["ANO"].max()
        )
    },

    "metricas": {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": float(macro_f1),
        "precision_con_muertos": float(
            precision_muertos
        ),
        "recall_con_muertos": float(
            recall_muertos
        ),
        "f1_con_muertos": float(
            f1_muertos
        ),
        "predicciones_con_muertos": int(
            predicciones_muertos
        )
    },

    "configuracion": {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": PESOS_CLASES,
        "random_state": 42
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
        indent=4,
        ensure_ascii=False
    )

print(
    "\nMétricas guardadas en:"
)

print(RUTA_METRICAS)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("PROCESO V9 FINALIZADO")
print("=" * 70)