import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_PATH = "data/processed/siniestros_modelo.csv"
MODEL_PATH = "models/modelo_v6.pkl"

OUTPUT_CSV = "models/importancia_variables_v6.csv"
OUTPUT_JSON = "models/importancia_variables_v6.json"


print("=" * 70)
print("ANÁLISIS DE IMPORTANCIA - RANDOM FOREST V6")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset: {df.shape[0]:,} registros x {df.shape[1]} columnas")


# ============================================================
# VARIABLES V6
# ============================================================

features = [
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
    "NUM_MOTOS_TOTAL",
]

target = "GRAVEDAD"


# ============================================================
# LIMPIEZA
# ============================================================

df = df.drop_duplicates().copy()

X = df[features]
y = df[target]


# ============================================================
# CARGAR MODELO
# ============================================================

print("\nCargando modelo V6...")

pipeline = joblib.load(MODEL_PATH)

print("Modelo cargado correctamente.")


# ============================================================
# IMPORTANCIA DEL RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANCIA DE VARIABLES")
print("=" * 70)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# Obtener nombres después del OneHotEncoder
feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_


importance_df = pd.DataFrame({
    "variable_transformada": feature_names,
    "importancia": importances
})

importance_df = importance_df.sort_values(
    "importancia",
    ascending=False
).reset_index(drop=True)


# ============================================================
# AGRUPAR IMPORTANCIA POR VARIABLE ORIGINAL
# ============================================================

def obtener_variable_original(nombre):

    nombre = nombre.replace("cat__", "")
    nombre = nombre.replace("num__", "")

    # Las variables categóricas aparecen como:
    # CLASE_xxx
    # CODIGO_LOCALIDAD_xxx

    for variable in features:

        if nombre == variable:
            return variable

        if nombre.startswith(variable + "_"):
            return variable

    return nombre


importance_df["variable_original"] = (
    importance_df["variable_transformada"]
    .apply(obtener_variable_original)
)


grouped_importance = (
    importance_df
    .groupby("variable_original", as_index=False)["importancia"]
    .sum()
    .sort_values("importancia", ascending=False)
    .reset_index(drop=True)
)


# Porcentaje
grouped_importance["porcentaje"] = (
    grouped_importance["importancia"] /
    grouped_importance["importancia"].sum() * 100
)


# ============================================================
# RESULTADOS
# ============================================================

print("\nImportancia agrupada por variable original:")

for i, row in grouped_importance.iterrows():

    print(
        f"{i + 1:2d}. "
        f"{row['variable_original']:<30} "
        f"{row['porcentaje']:>7.2f}%"
    )


# ============================================================
# TOP 15
# ============================================================

print("\n" + "=" * 70)
print("TOP 15 VARIABLES")
print("=" * 70)

print(
    grouped_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# VARIABLES POTENCIALMENTE DELICADAS
# ============================================================

sensitive_variables = [
    "NUM_VEHICULOS_EN_FUGA",
    "NUM_ACTORES",
    "NUM_CONDUCTORES",
    "NUM_PEATONES",
    "NUM_VEHICULOS",
    "NUM_SERVICIOS",
    "NUM_MODALIDADES",
]

print("\n" + "=" * 70)
print("VARIABLES A REVISAR")
print("=" * 70)

for variable in sensitive_variables:

    fila = grouped_importance[
        grouped_importance["variable_original"] == variable
    ]

    if not fila.empty:

        importance = fila.iloc[0]["porcentaje"]

        print(
            f"{variable:<30} {importance:>7.2f}%"
        )


# ============================================================
# GUARDAR CSV
# ============================================================

os.makedirs("models", exist_ok=True)

grouped_importance.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nImportancia guardada en:")
print(OUTPUT_CSV)


# ============================================================
# GUARDAR JSON
# ============================================================

importance_dict = {}

for _, row in grouped_importance.iterrows():

    importance_dict[row["variable_original"]] = {
        "importancia": float(row["importancia"]),
        "porcentaje": float(row["porcentaje"])
    }


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        importance_dict,
        f,
        indent=4,
        ensure_ascii=False
    )


print(f"JSON guardado en:")
print(OUTPUT_JSON)


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

print("\nLas 10 variables más importantes:")

for i, row in grouped_importance.head(10).iterrows():

    print(
        f"{i + 1}. "
        f"{row['variable_original']} "
        f"({row['porcentaje']:.2f}%)"
    )

print("\n" + "=" * 70)
print("ANÁLISIS V6 FINALIZADO")
print("=" * 70)