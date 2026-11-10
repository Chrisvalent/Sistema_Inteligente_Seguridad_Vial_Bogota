
import joblib
import pandas as pd

RUTA_MODELO = "models/modelo_v8b.pkl"
RUTA_SALIDA = "models/importancia_variables_v8b.csv"

print("=" * 70)
print("AUDITORIA DE IMPORTANCIA DE VARIABLES V8B")
print("=" * 70)

print("\nCargando modelo V8B...")
pipeline = joblib.load(RUTA_MODELO)

preprocesamiento = pipeline.named_steps["preprocesamiento"]
clasificador = pipeline.named_steps["modelo"]

# Importancia de las características transformadas
importancias = clasificador.feature_importances_

nombres_transformados = (
    preprocesamiento.get_feature_names_out()
)

print("\nCaracterísticas transformadas:", len(nombres_transformados))
print("Importancias disponibles:", len(importancias))

if len(nombres_transformados) != len(importancias):
    raise ValueError(
        "No coincide el número de características con las importancias."
    )

# Obtener las 34 variables originales
variables_originales = list(preprocesamiento.feature_names_in_)

# Relacionar cada característica transformada con su variable original
relaciones = []

for nombre in nombres_transformados:

    nombre = str(nombre)

    if "__" not in nombre:
        raise ValueError(
            f"No se reconoce el nombre transformado: {nombre}"
        )

    transformador, nombre_interno = nombre.split("__", 1)

    if transformador == "numericas":
        variable = nombre_interno

    elif transformador == "categoricas":

        coincidencias = [
            col for col in variables_originales
            if nombre_interno.startswith(str(col) + "_")
        ]

        if not coincidencias:
            raise ValueError(
                f"No se pudo identificar la variable original: {nombre}"
            )

        # Elegir la coincidencia más larga
        variable = max(coincidencias, key=len)

    else:
        raise ValueError(
            f"Transformador desconocido: {transformador}"
        )

    relaciones.append(variable)

# Construir tabla de importancias
detalle = pd.DataFrame({
    "variable_transformada": nombres_transformados,
    "variable_original": relaciones,
    "importancia": importancias
})

# Agrupar importancias por variable original
resultado = (
    detalle
    .groupby("variable_original", as_index=False)["importancia"]
    .sum()
    .sort_values("importancia", ascending=False)
    .reset_index(drop=True)
)

resultado["porcentaje"] = resultado["importancia"] * 100
resultado["ranking"] = range(1, len(resultado) + 1)

resultado = resultado[
    ["ranking", "variable_original", "importancia", "porcentaje"]
]

print("\n" + "=" * 70)
print("IMPORTANCIA DE LAS VARIABLES ORIGINALES")
print("=" * 70)

print(resultado.to_string(index=False))

print("\nTotal de variables originales:", len(resultado))
print(
    "Suma de importancias:",
    round(resultado["importancia"].sum(), 6)
)

# Guardar resultados
resultado.to_csv(
    RUTA_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:")
print(RUTA_SALIDA)

print("\nAuditoría terminada.")
