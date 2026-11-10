import pandas as pd
import os
import joblib

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

from imblearn.over_sampling import RandomOverSampler


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATASET = "data/processed/siniestros_limpios.csv"
RUTA_MODELO = "models/random_forest_v3.pkl"

CLASES = [
    "SOLO DANOS",
    "CON HERIDOS",
    "CON MUERTOS"
]


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST V3")
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
# 3. VARIABLES
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
# 4. DIVISIÓN TRAIN / TEST
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

print(f"\nEntrenamiento original: {len(X_train):,}")
print(f"Prueba:                 {len(X_test):,}")

print("\nDistribución original del entrenamiento:")

print(y_train.value_counts())


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
# 6. TRANSFORMAR TRAIN
# ============================================================

print("\nTransformando datos de entrenamiento...")

X_train_transformado = preprocesador.fit_transform(X_train)

print("Transformación completada.")

print(
    f"Dimensiones después del preprocesamiento: "
    f"{X_train_transformado.shape}"
)


# ============================================================
# 7. RANDOM OVERSAMPLER
# ============================================================

print("\n" + "=" * 70)
print("BALANCEO DEL CONJUNTO DE ENTRENAMIENTO")
print("=" * 70)

print("\nAplicando RandomOverSampler...")

oversampler = RandomOverSampler(
    random_state=42
)

X_train_balanceado, y_train_balanceado = oversampler.fit_resample(
    X_train_transformado,
    y_train
)

print("\nDistribución ANTES del balanceo:")

print(y_train.value_counts())

print("\nDistribución DESPUÉS del balanceo:")

print(y_train_balanceado.value_counts())

print(
    f"\nRegistros originales de entrenamiento: "
    f"{len(X_train):,}"
)

print(
    f"Registros después del balanceo: "
    f"{len(X_train_balanceado):,}"
)


# ============================================================
# 8. RANDOM FOREST V3
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN RANDOM FOREST V3")
print("=" * 70)

modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

print("\nParámetros:")
print("- n_estimators: 300")
print("- max_depth: None")
print("- min_samples_leaf: 2")
print("- max_features: sqrt")
print("- class_weight: None")
print("- balanceo: RandomOverSampler")
print("- random_state: 42")
print("- n_jobs: -1")


# ============================================================
# 9. ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO")
print("=" * 70)

print("\nEntrenando Random Forest V3...")
print("Esto puede tardar algunos minutos.")

modelo.fit(
    X_train_balanceado,
    y_train_balanceado
)

print("\nEntrenamiento completado.")


# ============================================================
# 10. TRANSFORMAR TEST
# ============================================================

print("\nTransformando conjunto de prueba...")

X_test_transformado = preprocesador.transform(
    X_test
)

print("Transformación del conjunto de prueba completada.")


# ============================================================
# 11. PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_pred = modelo.predict(
    X_test_transformado
)


# ============================================================
# 12. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 13. RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS RANDOM FOREST V3")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 14. CLASSIFICATION REPORT
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
# 15. MATRIZ DE CONFUSIÓN
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
# 16. MÉTRICAS CON MUERTOS
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
# 17. MÉTRICAS GLOBALES
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
# 18. GUARDAR MODELO + PREPROCESADOR
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO MODELO")
print("=" * 70)

os.makedirs(
    os.path.dirname(RUTA_MODELO),
    exist_ok=True
)

objeto_modelo = {
    "preprocesador": preprocesador,
    "modelo": modelo
}

joblib.dump(
    objeto_modelo,
    RUTA_MODELO
)

print("\nModelo guardado en:")
print(RUTA_MODELO)


# ============================================================
# 19. FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO FINALIZADO")
print("=" * 70)