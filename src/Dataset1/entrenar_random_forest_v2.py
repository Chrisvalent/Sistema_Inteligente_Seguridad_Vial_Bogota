import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATASET = "data/processed/siniestros_limpios.csv"
RUTA_MODELO = "models/random_forest_v2.pkl"

CLASES = [
    "SOLO DANOS",
    "CON HERIDOS",
    "CON MUERTOS"
]


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V2")
print("=" * 70)


# ============================================================
# 1. CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATASET)

print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 2. VARIABLE OBJETIVO
# ============================================================

X = df.drop(columns=["GRAVEDAD"])
y = df["GRAVEDAD"]

print("\nVariable objetivo:")
print(y.value_counts())


# ============================================================
# 3. DEFINIR VARIABLES
# ============================================================

columnas_categoricas = [
    "CLASE_ACC",
    "LOCALIDAD"
]

columnas_numericas = [
    "ANO_OCURRENCIA_ACC",
    "MES",
    "DIA",
    "HORA",
    "DIA_SEMANA",
    "ES_FIN_DE_SEMANA",
    "LATITUD",
    "LONGITUD",
    "CIV",
    "CIV_FALTANTE",
    "PK_CALZADA",
    "PK_CALZADA_FALTANTE"
]

print("\nVariables categóricas:")

for columna in columnas_categoricas:
    print(f"- {columna}")

print("\nVariables numéricas:")

for columna in columnas_numericas:
    print(f"- {columna}")


# ============================================================
# 4. DIVISIÓN ENTRENAMIENTO / PRUEBA
# ============================================================

print("\n" + "=" * 70)
print("DIVISIÓN DE DATOS")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nEntrenamiento: {len(X_train):,}")
print(f"Prueba:        {len(X_test):,}")

print("\nDistribución en entrenamiento:")

print(
    y_train.value_counts()
)

print("\nDistribución en prueba:")

print(
    y_test.value_counts()
)


# ============================================================
# 5. PREPROCESAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            columnas_categoricas
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 6. RANDOM FOREST V2
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN RANDOM FOREST V2")
print("=" * 70)

modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\nParámetros:")
print("- n_estimators: 300")
print("- max_depth: None")
print("- min_samples_leaf: 2")
print("- max_features: sqrt")
print("- class_weight: balanced_subsample")
print("- random_state: 42")
print("- n_jobs: -1")


# ============================================================
# 7. CREAR PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("modelo", modelo)
    ]
)


# ============================================================
# 8. ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print("\nEntrenando Random Forest V2...")
print("Esto puede tardar algunos minutos.")

pipeline.fit(X_train, y_train)

print("\nEntrenamiento completado.")


# ============================================================
# 9. PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_pred = pipeline.predict(X_test)


# ============================================================
# 10. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 11. RESULTADOS GENERALES
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS RANDOM FOREST V2")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

reporte = classification_report(
    y_test,
    y_pred,
    labels=CLASES,
    digits=4
)

print(reporte)


# ============================================================
# 13. MATRIZ DE CONFUSIÓN
# ============================================================

print("===== MATRIZ DE CONFUSIÓN =====")

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASES
)

print("\n                  Predicción")
print("                SD      H       M")
print("-------------------------------------")

for i, clase in enumerate(CLASES):

    valores = matriz[i]

    if clase == "SOLO DANOS":
        nombre = "Real SD"
    elif clase == "CON HERIDOS":
        nombre = "Real H "
    else:
        nombre = "Real M "

    print(
        f"{nombre}      "
        f"{valores[0]:5d}   "
        f"{valores[1]:5d}   "
        f"{valores[2]:5d}"
    )


# ============================================================
# 14. MÉTRICAS ESPECÍFICAS
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS DE LA CLASE CON MUERTOS")
print("=" * 70)

reporte_dict = classification_report(
    y_test,
    y_pred,
    labels=CLASES,
    output_dict=True
)

metricas_muertos = reporte_dict["CON MUERTOS"]

print(
    f"\nPrecision CON MUERTOS: "
    f"{metricas_muertos['precision']:.4f}"
)

print(
    f"Recall CON MUERTOS:    "
    f"{metricas_muertos['recall']:.4f}"
)

print(
    f"F1 CON MUERTOS:       "
    f"{metricas_muertos['f1-score']:.4f}"
)

print(
    f"Soporte CON MUERTOS:  "
    f"{int(metricas_muertos['support'])}"
)


# ============================================================
# 15. MÉTRICAS MACRO
# ============================================================

print("\n" + "=" * 70)
print("MÉTRICAS GLOBALES")
print("=" * 70)

print(
    f"\nMacro Precision: "
    f"{reporte_dict['macro avg']['precision']:.4f}"
)

print(
    f"Macro Recall:    "
    f"{reporte_dict['macro avg']['recall']:.4f}"
)

print(
    f"Macro F1:        "
    f"{reporte_dict['macro avg']['f1-score']:.4f}"
)

print(
    f"Weighted F1:     "
    f"{reporte_dict['weighted avg']['f1-score']:.4f}"
)


# ============================================================
# 16. GUARDAR MODELO
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO MODELO")
print("=" * 70)

os.makedirs(
    os.path.dirname(RUTA_MODELO),
    exist_ok=True
)

joblib.dump(
    pipeline,
    RUTA_MODELO
)

print(f"\nModelo guardado en:")
print(RUTA_MODELO)


# ============================================================
# 17. FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO FINALIZADO")
print("=" * 70)