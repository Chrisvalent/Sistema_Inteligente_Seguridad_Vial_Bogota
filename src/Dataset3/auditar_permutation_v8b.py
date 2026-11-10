
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, balanced_accuracy_score

RUTA_MODELO = "models/modelo_v8b.pkl"
RUTA_DATASET = "data/processed/siniestros_modelo.csv"
RUTA_SALIDA = "models/permutation_importance_v8b.csv"

print("=" * 70)
print("AUDITORIA PERMUTATION IMPORTANCE - V8B")
print("=" * 70)

# 1. Cargar modelo y dataset
print("\nCargando modelo y dataset...")

modelo = joblib.load(RUTA_MODELO)
df = pd.read_csv(RUTA_DATASET, low_memory=False)

# Recuperar exactamente las variables utilizadas por el modelo
variables = list(modelo.feature_names_in_)

# Reproducir la eliminación de duplicados del entrenamiento
df = df.drop_duplicates().copy()

X = df[variables]
y = df["GRAVEDAD"]

print("Registros después de eliminar duplicados:", len(df))
print("Variables:", len(variables))

# 2. Reconstruir las particiones originales
print("\nReconstruyendo particiones originales...")

X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_validacion, y_train, y_validacion = train_test_split(
    X_desarrollo,
    y_desarrollo,
    test_size=0.20,
    stratify=y_desarrollo,
    random_state=42
)

print("Entrenamiento:", len(X_train))
print("Validación:", len(X_validacion))
print("Prueba:", len(X_test))

# 3. Evaluación base del pipeline
print("\nCalculando rendimiento base...")

predicciones = modelo.predict(X_test)

f1_base = f1_score(
    y_test,
    predicciones,
    average="macro"
)

balanced_base = balanced_accuracy_score(
    y_test,
    predicciones
)

print("F1 macro base:", round(f1_base, 6))
print("Balanced accuracy base:", round(balanced_base, 6))

# 4. Calcular Permutation Importance
print("\nCalculando Permutation Importance...")
print("Métrica: F1 macro")
print("Repeticiones por variable: 3")
print("Este proceso puede tardar varios minutos.")

resultado = permutation_importance(
    modelo,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

# 5. Organizar resultados
tabla = pd.DataFrame({
    "variable": variables,
    "importancia_media": resultado.importances_mean,
    "desviacion_estandar": resultado.importances_std
})

tabla = tabla.sort_values(
    "importancia_media",
    ascending=False
).reset_index(drop=True)

tabla["ranking"] = range(1, len(tabla) + 1)

tabla = tabla[
    [
        "ranking",
        "variable",
        "importancia_media",
        "desviacion_estandar"
    ]
]

# 6. Mostrar resultados
print("\n" + "=" * 70)
print("RESULTADOS PERMUTATION IMPORTANCE")
print("=" * 70)

print(tabla.to_string(index=False))

# 7. Guardar archivo
tabla.to_csv(
    RUTA_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("AUDITORIA FINALIZADA")
print("=" * 70)

print("F1 macro base:", round(f1_base, 6))
print("Balanced accuracy base:", round(balanced_base, 6))
print("Archivo generado:", RUTA_SALIDA)

print("\nInterpretación:")
print("- Importancia positiva: al alterar la variable, el F1 macro disminuye.")
print("- Importancia cercana a cero: efecto pequeño en esta evaluación.")
print("- Importancia negativa: alterar la variable mejoró el resultado.")
print("- No representa causalidad ni demuestra capacidad preventiva.")
