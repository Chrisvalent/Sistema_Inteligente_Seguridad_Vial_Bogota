import pandas as pd

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

# ============================================================
# 1. CARGAR DATASET
# ============================================================

RUTA_DATASET = "data/processed/siniestros_limpios.csv"

print("=" * 70)
print("ENTRENAMIENTO - RANDOM FOREST")
print("=" * 70)

print("\nCargando dataset...")

df = pd.read_csv(RUTA_DATASET)

print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 2. SEPARAR VARIABLE OBJETIVO Y PREDICTORAS
# ============================================================

X = df.drop(columns=["GRAVEDAD"])
y = df["GRAVEDAD"]

print("\nVariable objetivo:")
print(y.value_counts())


# ============================================================
# 3. IDENTIFICAR VARIABLES
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
    print("-", columna)

print("\nVariables numéricas:")
for columna in columnas_numericas:
    print("-", columna)


# ============================================================
# 4. DIVIDIR TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDivisión de datos:")
print(f"Entrenamiento: {len(X_train):,}")
print(f"Prueba:        {len(X_test):,}")


# ============================================================
# 5. PREPROCESAMIENTO
# ============================================================

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            columnas_categoricas
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 6. MODELO RANDOM FOREST
# ============================================================

modelo = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. PIPELINE
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

print("\nEntrenando Random Forest...")
print("Esto puede tardar algunos minutos.")

pipeline.fit(X_train, y_train)

print("Entrenamiento completado.")


# ============================================================
# 9. PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_pred = pipeline.predict(X_test)


# ============================================================
# 10. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================================================
# 12. MATRIZ DE CONFUSIÓN
# ============================================================

print("\n===== MATRIZ DE CONFUSIÓN =====")

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "SOLO DANOS",
        "CON HERIDOS",
        "CON MUERTOS"
    ]
)

print("\n                  Predicción")
print("                SD      H       M")
print("-------------------------------------")
print(
    f"Real SD       {matriz[0][0]:6d} "
    f"{matriz[0][1]:6d} "
    f"{matriz[0][2]:6d}"
)
print(
    f"Real H        {matriz[1][0]:6d} "
    f"{matriz[1][1]:6d} "
    f"{matriz[1][2]:6d}"
)
print(
    f"Real M        {matriz[2][0]:6d} "
    f"{matriz[2][1]:6d} "
    f"{matriz[2][2]:6d}"
)

print("\nProceso finalizado.")