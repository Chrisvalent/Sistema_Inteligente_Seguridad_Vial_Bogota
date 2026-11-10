import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
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
RUTA_MODELO = "models/modelo_v8b.pkl"
RUTA_METRICAS = "models/metricas_modelo_v8b.json"
RUTA_UMBRALES = "models/umbrales_modelo_v8b.csv"

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

CLASES = ["Con Heridos", "Con Muertos", "Solo Daños"]


# ============================================================
# FUNCIÓN PARA APLICAR UMBRAL
# ============================================================

def aplicar_umbral(probabilidades, clases, umbral):
    """
    Predicción normal por máxima probabilidad,
    pero fuerza 'Con Muertos' cuando su probabilidad
    supera el umbral.
    """

    indice_muertos = list(clases).index("Con Muertos")

    predicciones_base = clases[np.argmax(probabilidades, axis=1)]

    predicciones = predicciones_base.copy()

    mascara_muertos = probabilidades[:, indice_muertos] >= umbral
    predicciones[mascara_muertos] = "Con Muertos"

    return predicciones


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V8B")
print("VALIDACIÓN DE UMBRAL + PRUEBA INTACTA")
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
# VARIABLE OBJETIVO
# ============================================================

print("\n" + "=" * 70)
print("VARIABLE OBJETIVO")
print("=" * 70)

print(df[VARIABLE_OBJETIVO].value_counts())

print("\nPorcentajes:")
print(
    df[VARIABLE_OBJETIVO]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# X / y
# ============================================================

X = df[VARIABLES].copy()
y = df[VARIABLE_OBJETIVO].copy()


# ============================================================
# DIVISIÓN 64 / 16 / 20
# ============================================================

print("\n" + "=" * 70)
print("DIVISIÓN DE DATOS")
print("=" * 70)

# Primero: 80% desarrollo / 20% prueba
X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Segundo: dentro del 80%, separar 20% para validación
# Esto produce aproximadamente:
# 64% entrenamiento
# 16% validación
# 20% prueba

X_train, X_validacion, y_train, y_validacion = train_test_split(
    X_desarrollo,
    y_desarrollo,
    test_size=0.20,
    stratify=y_desarrollo,
    random_state=42
)

print(f"Entrenamiento: {len(X_train):,}")
print(f"Validación:    {len(X_validacion):,}")
print(f"Prueba:        {len(X_test):,}")


print("\nDistribución entrenamiento:")
print(y_train.value_counts())

print("\nDistribución validación:")
print(y_validacion.value_counts())

print("\nDistribución prueba:")
print(y_test.value_counts())


# ============================================================
# PREPROCESAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

transformador_numerico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

transformador_categorico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
print("CONFIGURACIÓN RANDOM FOREST V8B")
print("=" * 70)

pesos_clases = {
    "Con Heridos": 1.0,
    "Con Muertos": 5.0,
    "Solo Daños": 1.0
}

print("- n_estimators: 400")
print("- max_depth: None")
print("- min_samples_leaf: 2")
print("- max_features: sqrt")
print(f"- class_weight: {pesos_clases}")
print("- random_state: 42")
print("- n_jobs: -1")


modelo_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=pesos_clases,
    random_state=42,
    n_jobs=-1
)


modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesamiento),
        ("modelo", modelo_rf)
    ]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print("\nEntrenando Random Forest V8B...")
print("Esto puede tardar algunos minutos.")

modelo.fit(X_train, y_train)

print("\nEntrenamiento completado.")


# ============================================================
# PROBABILIDADES VALIDACIÓN
# ============================================================

print("\nGenerando probabilidades de validación...")

prob_validacion = modelo.predict_proba(X_validacion)

print("Probabilidades de validación generadas correctamente.")


# ============================================================
# BÚSQUEDA DEL UMBRAL EN VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("SELECCIÓN DE UMBRAL - VALIDACIÓN")
print("=" * 70)

umbrales = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.075,
    0.05
]

resultados_umbrales = []

for umbral in umbrales:

    pred_validacion = aplicar_umbral(
        prob_validacion,
        modelo.classes_,
        umbral
    )

    precision_muertos = precision_score(
        y_validacion,
        pred_validacion,
        labels=["Con Muertos"],
        average="macro",
        zero_division=0
    )

    recall_muertos = recall_score(
        y_validacion,
        pred_validacion,
        labels=["Con Muertos"],
        average="macro",
        zero_division=0
    )

    f1_muertos = f1_score(
        y_validacion,
        pred_validacion,
        labels=["Con Muertos"],
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_validacion,
        pred_validacion
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validacion,
        pred_validacion
    )

    macro_f1 = f1_score(
        y_validacion,
        pred_validacion,
        average="macro",
        zero_division=0
    )

    predicciones_muertos = np.sum(
        pred_validacion == "Con Muertos"
    )

    resultados_umbrales.append({
        "umbral": umbral,
        "precision_muertos": precision_muertos,
        "recall_muertos": recall_muertos,
        "f1_muertos": f1_muertos,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "predicciones_muertos": predicciones_muertos
    })


df_umbrales = pd.DataFrame(resultados_umbrales)


print("\n")
print(
    df_umbrales.to_string(
        index=False,
        formatters={
            "precision_muertos": "{:.4f}".format,
            "recall_muertos": "{:.4f}".format,
            "f1_muertos": "{:.4f}".format,
            "accuracy": "{:.4f}".format,
            "balanced_accuracy": "{:.4f}".format,
            "macro_f1": "{:.4f}".format
        }
    )
)


# ============================================================
# MEJOR UMBRAL
# ============================================================

mejor = df_umbrales.sort_values(
    by=[
        "f1_muertos",
        "recall_muertos",
        "precision_muertos"
    ],
    ascending=False
).iloc[0]

mejor_umbral = float(mejor["umbral"])


print("\n" + "=" * 70)
print("MEJOR UMBRAL - VALIDACIÓN")
print("=" * 70)

print(f"Umbral seleccionado:       {mejor_umbral:.3f}")
print(
    f"Precision CON MUERTOS:     "
    f"{mejor['precision_muertos']:.4f}"
)
print(
    f"Recall CON MUERTOS:        "
    f"{mejor['recall_muertos']:.4f}"
)
print(
    f"F1 CON MUERTOS:            "
    f"{mejor['f1_muertos']:.4f}"
)
print(f"Accuracy:                  {mejor['accuracy']:.4f}")
print(
    f"Balanced Accuracy:         "
    f"{mejor['balanced_accuracy']:.4f}"
)
print(f"Macro F1:                  {mejor['macro_f1']:.4f}")
print(
    f"Predicciones CON MUERTOS:  "
    f"{int(mejor['predicciones_muertos'])}"
)


# ============================================================
# EVALUACIÓN FINAL SOBRE TEST
# ============================================================

print("\n" + "=" * 70)
print("EVALUACIÓN FINAL - CONJUNTO DE PRUEBA")
print("=" * 70)

print("\nIMPORTANTE:")
print(
    "El conjunto de prueba NO fue utilizado para seleccionar "
    "el umbral."
)

print("\nGenerando probabilidades de prueba...")

prob_test = modelo.predict_proba(X_test)

print("Probabilidades de prueba generadas correctamente.")


pred_test = aplicar_umbral(
    prob_test,
    modelo.classes_,
    mejor_umbral
)


# ============================================================
# MÉTRICAS FINALES
# ============================================================

accuracy_test = accuracy_score(
    y_test,
    pred_test
)

balanced_accuracy_test = balanced_accuracy_score(
    y_test,
    pred_test
)

macro_f1_test = f1_score(
    y_test,
    pred_test,
    average="macro",
    zero_division=0
)

precision_muertos_test = precision_score(
    y_test,
    pred_test,
    labels=["Con Muertos"],
    average="macro",
    zero_division=0
)

recall_muertos_test = recall_score(
    y_test,
    pred_test,
    labels=["Con Muertos"],
    average="macro",
    zero_division=0
)

f1_muertos_test = f1_score(
    y_test,
    pred_test,
    labels=["Con Muertos"],
    average="macro",
    zero_division=0
)

predicciones_muertos_test = np.sum(
    pred_test == "Con Muertos"
)


# ============================================================
# RESULTADOS FINALES
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - V8B TEST")
print("=" * 70)

print(
    classification_report(
        y_test,
        pred_test,
        labels=CLASES,
        zero_division=0
    )
)


print("=" * 70)
print("MATRIZ DE CONFUSIÓN - V8B TEST")
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


print("\n" + "=" * 70)
print("RESUMEN FINAL V8B")
print("=" * 70)

print(f"Registros utilizados:       {len(df):,}")
print(f"Variables predictoras:      {len(VARIABLES)}")
print(f"Umbral seleccionado:        {mejor_umbral:.3f}")

print(f"\nAccuracy:                   {accuracy_test:.4f}")
print(
    f"Balanced Accuracy:          "
    f"{balanced_accuracy_test:.4f}"
)
print(f"Macro F1:                   {macro_f1_test:.4f}")

print(
    f"\nPrecision Con Muertos:      "
    f"{precision_muertos_test:.4f}"
)
print(
    f"Recall Con Muertos:         "
    f"{recall_muertos_test:.4f}"
)
print(
    f"F1 Con Muertos:             "
    f"{f1_muertos_test:.4f}"
)

print(
    f"\nPredicciones Con Muertos:   "
    f"{predicciones_muertos_test}"
)


# ============================================================
# GUARDAR MODELO
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO MODELO")
print("=" * 70)

import joblib

joblib.dump(
    modelo,
    RUTA_MODELO
)

print(f"Modelo guardado en:")
print(RUTA_MODELO)


# ============================================================
# GUARDAR TABLA DE UMBRALES
# ============================================================

df_umbrales.to_csv(
    RUTA_UMBRALES,
    index=False
)

print("\nTabla de umbrales guardada en:")
print(RUTA_UMBRALES)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = {
    "modelo": "Random Forest V8B",
    "registros": int(len(df)),
    "variables_predictoras": int(len(VARIABLES)),
    "entrenamiento": int(len(X_train)),
    "validacion": int(len(X_validacion)),
    "prueba": int(len(X_test)),
    "umbral_seleccionado": mejor_umbral,

    "validacion": {
        "precision_con_muertos": float(mejor["precision_muertos"]),
        "recall_con_muertos": float(mejor["recall_muertos"]),
        "f1_con_muertos": float(mejor["f1_muertos"]),
        "accuracy": float(mejor["accuracy"]),
        "balanced_accuracy": float(mejor["balanced_accuracy"]),
        "macro_f1": float(mejor["macro_f1"])
    },

    "prueba": {
        "precision_con_muertos": float(
            precision_muertos_test
        ),
        "recall_con_muertos": float(
            recall_muertos_test
        ),
        "f1_con_muertos": float(
            f1_muertos_test
        ),
        "accuracy": float(
            accuracy_test
        ),
        "balanced_accuracy": float(
            balanced_accuracy_test
        ),
        "macro_f1": float(
            macro_f1_test
        ),
        "predicciones_con_muertos": int(
            predicciones_muertos_test
        )
    },

    "configuracion": {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": pesos_clases,
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


print("\nMétricas guardadas en:")
print(RUTA_METRICAS)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("PROCESO V8B FINALIZADO")
print("=" * 70)