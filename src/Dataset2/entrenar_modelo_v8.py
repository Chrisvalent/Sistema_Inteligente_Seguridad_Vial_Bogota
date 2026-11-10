import os
import json
import joblib
import numpy as np
import pandas as pd

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
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATASET = "data/processed/siniestros_modelo.csv"
RUTA_MODELO = "models/modelo_v8.pkl"
RUTA_METRICAS = "models/metricas_modelo_v8.json"
RUTA_UMBRALES = "models/umbrales_modelo_v8.csv"

RANDOM_STATE = 42

CLASE_MUERTOS = "Con Muertos"

# Pesos utilizados en V7
PESOS_CLASE = {
    "Con Heridos": 1.0,
    "Con Muertos": 5.0,
    "Solo Daños": 1.0
}


# ============================================================
# VARIABLES
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

VARIABLES = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V8")
print("AJUSTE DE UMBRAL PARA CON MUERTOS")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATASET)

print(f"Dataset original: {df.shape}")


# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

duplicados = df.duplicated().sum()

print("\n" + "=" * 70)
print("DUPLICADOS")
print("=" * 70)

print(f"Duplicados encontrados: {duplicados}")

df = df.drop_duplicates().copy()

print(f"Registros finales: {len(df):,}")


# ============================================================
# VARIABLE OBJETIVO
# ============================================================

TARGET = "GRAVEDAD"

X = df[VARIABLES].copy()
y = df[TARGET].copy()

print("\n" + "=" * 70)
print("VARIABLE OBJETIVO")
print("=" * 70)

print(y.value_counts())

print("\nPorcentajes:")
print((y.value_counts(normalize=True) * 100).round(2))


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

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            VARIABLES_CATEGORICAS
        ),
        (
            "numericas",
            "passthrough",
            VARIABLES_NUMERICAS
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN RANDOM FOREST V8")
print("=" * 70)

print("- n_estimators: 400")
print("- max_depth: None")
print("- min_samples_leaf: 2")
print("- max_features: sqrt")
print("- class_weight:", PESOS_CLASE)
print("- random_state: 42")
print("- n_jobs: -1")


modelo = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=PESOS_CLASE,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("modelo", modelo)
    ]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print("\nEntrenando Random Forest V8...")
print("Esto puede tardar algunos minutos.")

pipeline.fit(X_train, y_train)

print("\nEntrenamiento completado.")


# ============================================================
# PROBABILIDADES
# ============================================================

print("\nGenerando probabilidades...")

probabilidades = pipeline.predict_proba(X_test)

modelo_rf = pipeline.named_steps["modelo"]

clases = list(modelo_rf.classes_)

indice_muertos = clases.index(CLASE_MUERTOS)

prob_muertos = probabilidades[:, indice_muertos]

print("Probabilidades generadas correctamente.")


# ============================================================
# FUNCIÓN PARA APLICAR UMBRAL
# ============================================================

def predecir_con_umbral(
    probabilidades,
    clases,
    prob_muertos,
    umbral
):
    """
    Mantiene la predicción normal del modelo,
    pero fuerza 'Con Muertos' cuando su probabilidad
    supera el umbral indicado.
    """

    predicciones = np.array(clases)[
        np.argmax(probabilidades, axis=1)
    ]

    mascara_muertos = prob_muertos >= umbral

    predicciones[mascara_muertos] = CLASE_MUERTOS

    return predicciones


# ============================================================
# UMBRALES A PROBAR
# ============================================================

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


# ============================================================
# EVALUACIÓN DE UMBRALES
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS DE UMBRALES")
print("=" * 70)

resultados = []

for umbral in umbrales:

    y_pred_umbral = predecir_con_umbral(
        probabilidades,
        clases,
        prob_muertos,
        umbral
    )

    precision_muertos = precision_score(
        y_test,
        y_pred_umbral,
        labels=[CLASE_MUERTOS],
        average="macro",
        zero_division=0
    )

    recall_muertos = recall_score(
        y_test,
        y_pred_umbral,
        labels=[CLASE_MUERTOS],
        average="macro",
        zero_division=0
    )

    f1_muertos = f1_score(
        y_test,
        y_pred_umbral,
        labels=[CLASE_MUERTOS],
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred_umbral
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred_umbral
    )

    macro_f1 = f1_score(
        y_test,
        y_pred_umbral,
        average="macro",
        zero_division=0
    )

    numero_predichos_muertos = int(
        np.sum(y_pred_umbral == CLASE_MUERTOS)
    )

    resultados.append({
        "umbral": umbral,
        "precision_muertos": precision_muertos,
        "recall_muertos": recall_muertos,
        "f1_muertos": f1_muertos,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "predicciones_muertos": numero_predichos_muertos
    })


df_resultados = pd.DataFrame(resultados)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n")
print(
    df_resultados.to_string(
        index=False,
        formatters={
            "umbral": "{:.3f}".format,
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
# SELECCIONAR MEJOR UMBRAL
# ============================================================

# Criterio principal:
# maximizar F1 de CON MUERTOS.
#
# En caso de empate:
# 1. mayor recall
# 2. mayor precision
# 3. mayor balanced accuracy

df_resultados_ordenado = df_resultados.sort_values(
    by=[
        "f1_muertos",
        "recall_muertos",
        "precision_muertos",
        "balanced_accuracy"
    ],
    ascending=False
)

mejor = df_resultados_ordenado.iloc[0]

mejor_umbral = float(mejor["umbral"])


# ============================================================
# RESULTADO DEL MEJOR UMBRAL
# ============================================================

print("\n" + "=" * 70)
print("MEJOR UMBRAL")
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
print(
    f"Accuracy:                  "
    f"{mejor['accuracy']:.4f}"
)
print(
    f"Balanced Accuracy:         "
    f"{mejor['balanced_accuracy']:.4f}"
)
print(
    f"Macro F1:                  "
    f"{mejor['macro_f1']:.4f}"
)
print(
    f"Predicciones CON MUERTOS:  "
    f"{int(mejor['predicciones_muertos']):,}"
)


# ============================================================
# PREDICCIÓN FINAL CON EL MEJOR UMBRAL
# ============================================================

y_pred_final = predecir_con_umbral(
    probabilidades,
    clases,
    prob_muertos,
    mejor_umbral
)


# ============================================================
# REPORTE FINAL
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - V8")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred_final,
        zero_division=0
    )
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

print("=" * 70)
print("MATRIZ DE CONFUSIÓN - V8")
print("=" * 70)

matriz = confusion_matrix(
    y_test,
    y_pred_final,
    labels=clases
)

df_matriz = pd.DataFrame(
    matriz,
    index=clases,
    columns=clases
)

print(df_matriz)


# ============================================================
# MÉTRICAS FINALES
# ============================================================

accuracy_final = accuracy_score(
    y_test,
    y_pred_final
)

balanced_accuracy_final = balanced_accuracy_score(
    y_test,
    y_pred_final
)

macro_precision_final = precision_score(
    y_test,
    y_pred_final,
    average="macro",
    zero_division=0
)

macro_recall_final = recall_score(
    y_test,
    y_pred_final,
    average="macro",
    zero_division=0
)

macro_f1_final = f1_score(
    y_test,
    y_pred_final,
    average="macro",
    zero_division=0
)

weighted_f1_final = f1_score(
    y_test,
    y_pred_final,
    average="weighted",
    zero_division=0
)

precision_muertos_final = precision_score(
    y_test,
    y_pred_final,
    labels=[CLASE_MUERTOS],
    average="macro",
    zero_division=0
)

recall_muertos_final = recall_score(
    y_test,
    y_pred_final,
    labels=[CLASE_MUERTOS],
    average="macro",
    zero_division=0
)

f1_muertos_final = f1_score(
    y_test,
    y_pred_final,
    labels=[CLASE_MUERTOS],
    average="macro",
    zero_division=0
)


# ============================================================
# GUARDAR MODELO
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    RUTA_MODELO
)

print("\n" + "=" * 70)
print("GUARDANDO MODELO")
print("=" * 70)

print(f"Modelo guardado en:")
print(RUTA_MODELO)


# ============================================================
# GUARDAR TABLA DE UMBRALES
# ============================================================

df_resultados.to_csv(
    RUTA_UMBRALES,
    index=False,
    encoding="utf-8-sig"
)

print("\nTabla de umbrales guardada en:")
print(RUTA_UMBRALES)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

reporte = classification_report(
    y_test,
    y_pred_final,
    output_dict=True,
    zero_division=0
)

metricas = {
    "modelo": "Random Forest V8",
    "descripcion": (
        "Random Forest V7 con ajuste de umbral "
        "para la clase Con Muertos"
    ),

    "registros": int(len(df)),
    "variables_predictoras": int(len(VARIABLES)),

    "umbral_seleccionado": mejor_umbral,

    "accuracy": float(accuracy_final),
    "balanced_accuracy": float(balanced_accuracy_final),

    "macro_precision": float(
        macro_precision_final
    ),

    "macro_recall": float(
        macro_recall_final
    ),

    "macro_f1": float(
        macro_f1_final
    ),

    "weighted_f1": float(
        weighted_f1_final
    ),

    "precision_con_muertos": float(
        precision_muertos_final
    ),

    "recall_con_muertos": float(
        recall_muertos_final
    ),

    "f1_con_muertos": float(
        f1_muertos_final
    ),

    "pesos_clase": PESOS_CLASE,

    "reporte_clasificacion": reporte,

    "matriz_confusion": matriz.tolist(),

    "clases": clases
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
# RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL V8")
print("=" * 70)

print(f"Registros utilizados:       {len(df):,}")
print(f"Variables predictoras:      {len(VARIABLES)}")
print(f"Umbral seleccionado:        {mejor_umbral:.3f}")

print(f"\nAccuracy:                   {accuracy_final:.4f}")
print(
    f"Balanced Accuracy:          "
    f"{balanced_accuracy_final:.4f}"
)
print(
    f"Macro F1:                   "
    f"{macro_f1_final:.4f}"
)

print(
    f"\nPrecision Con Muertos:      "
    f"{precision_muertos_final:.4f}"
)
print(
    f"Recall Con Muertos:         "
    f"{recall_muertos_final:.4f}"
)
print(
    f"F1 Con Muertos:             "
    f"{f1_muertos_final:.4f}"
)

print("\nModelo:")
print(RUTA_MODELO)

print("\nMétricas:")
print(RUTA_METRICAS)

print("\nTabla de umbrales:")
print(RUTA_UMBRALES)

print("\n" + "=" * 70)
print("PROCESO V8 FINALIZADO")
print("=" * 70)