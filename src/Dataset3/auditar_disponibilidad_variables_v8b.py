
import joblib
import pandas as pd

RUTA_MODELO = "models/modelo_v8b.pkl"
RUTA_SALIDA = "models/auditoria_disponibilidad_v8b.csv"

print("=" * 75)
print("AUDITORIA DE DISPONIBILIDAD TEMPORAL DE VARIABLES V8B")
print("=" * 75)

modelo = joblib.load(RUTA_MODELO)

variables = list(modelo.feature_names_in_)

# Clasificación basada en la procedencia y construcción de las variables
clasificacion = {

    # Variables temporales:
    # Pueden conocerse antes si se proporciona una fecha y hora futuras.
    "ANO": (
        "Potencialmente previa",
        "Disponible si corresponde al año que se desea consultar."
    ),
    "MES": (
        "Potencialmente previa",
        "Disponible si corresponde al mes que se desea consultar."
    ),
    "DIA": (
        "Potencialmente previa",
        "Disponible si corresponde al día que se desea consultar."
    ),
    "DIA_SEMANA": (
        "Potencialmente previa",
        "Calculable a partir de una fecha futura conocida."
    ),
    "ES_FIN_DE_SEMANA": (
        "Potencialmente previa",
        "Calculable a partir de una fecha futura conocida."
    ),
    "HORA_NUM": (
        "Potencialmente previa",
        "Disponible si se proporciona anticipadamente la hora de consulta."
    ),
    "FRANJA_HORARIA": (
        "Potencialmente previa",
        "Calculable a partir de una hora futura conocida."
    ),

    # Ubicación:
    "CODIGO_LOCALIDAD": (
        "Potencialmente previa",
        "La localidad puede conocerse si se proporciona la ubicación de consulta."
    ),
    "DISENO_LUGAR": (
        "Disponibilidad por verificar",
        "Debe comprobarse si procede de un inventario vial previo o del reporte del accidente."
    ),

    # Características del accidente:
    "CLASE": (
        "Posterior al accidente",
        "Tipo de accidente registrado en la fuente SINIESTROS."
    ),
    "CHOQUE": (
        "Posterior al accidente",
        "Describe el elemento involucrado en el choque."
    ),
    "OBJETO_FIJO": (
        "Posterior al accidente",
        "Describe el objeto fijo involucrado; además presenta alta ausencia original."
    ),

    # Conteos de actores:
    "NUM_ACTORES": (
        "Posterior al accidente",
        "Conteo de participantes asociados al accidente registrado."
    ),
    "NUM_CONDUCTORES": (
        "Posterior al accidente",
        "Conteo de conductores registrados como participantes."
    ),
    "NUM_PEATONES": (
        "Posterior al accidente",
        "Conteo de peatones registrados como participantes."
    ),

    # Conteos de vehículos:
    "NUM_VEHICULOS": (
        "Posterior al accidente",
        "Conteo de vehículos asociados al accidente registrado."
    ),
    "NUM_MOTOS_TOTAL": (
        "Posterior al accidente",
        "Conteo agregado de motocicletas asociadas al accidente."
    ),
    "NUM_MOTOCICLETA": (
        "Posterior al accidente",
        "Conteo de motocicletas asociadas al accidente."
    ),
    "NUM_BICICLETA": (
        "Posterior al accidente",
        "Conteo de bicicletas asociadas al accidente."
    ),
    "NUM_AUTOMOVIL": (
        "Posterior al accidente",
        "Conteo de automóviles asociados al accidente."
    ),
    "NUM_SERVICIOS": (
        "Posterior al accidente",
        "Cantidad de categorías de servicio de los vehículos registrados."
    ),
    "NUM_MODALIDADES": (
        "Posterior al accidente",
        "Cantidad de modalidades de los vehículos registrados."
    ),
    "NUM_CLASES_VEHICULO": (
        "Posterior al accidente",
        "Cantidad de clases de vehículos asociados al accidente."
    ),
    "NUM_CAMIONETA": (
        "Posterior al accidente",
        "Conteo de camionetas asociadas al accidente."
    ),
    "NUM_CAMION_FURGON": (
        "Posterior al accidente",
        "Conteo de camiones o furgones asociados al accidente."
    ),
    "NUM_BUS": (
        "Posterior al accidente",
        "Conteo de buses asociados al accidente."
    ),
    "NUM_CAMPERO": (
        "Posterior al accidente",
        "Conteo de camperos asociados al accidente."
    ),
    "NUM_TRACTOCAMION": (
        "Posterior al accidente",
        "Conteo de tractocamiones asociados al accidente."
    ),
    "NUM_VEHICULOS_EN_FUGA": (
        "Posterior al accidente",
        "Se determina a partir del registro de vehículos involucrados."
    ),
    "NUM_VOLQUETA": (
        "Posterior al accidente",
        "Conteo de volquetas asociadas al accidente."
    ),
    "NUM_MICROBUS": (
        "Posterior al accidente",
        "Conteo de microbuses asociados al accidente."
    ),
    "NUM_BUSETA": (
        "Posterior al accidente",
        "Conteo de busetas asociadas al accidente."
    ),
    "NUM_BICITAXI": (
        "Posterior al accidente",
        "Conteo de bicitaxis asociados al accidente."
    ),
    "NUM_MOTOCARRO": (
        "Posterior al accidente",
        "Conteo de motocarros asociados al accidente."
    ),
}

# Verificar que todas las variables del modelo estén clasificadas
faltantes = [
    variable for variable in variables
    if variable not in clasificacion
]

if faltantes:
    raise ValueError(
        f"Hay variables sin clasificar: {faltantes}"
    )

# Crear matriz de auditoría
registros = []

for variable in variables:
    categoria, justificacion = clasificacion[variable]

    registros.append({
        "variable": variable,
        "clasificacion": categoria,
        "justificacion": justificacion
    })

resultado = pd.DataFrame(registros)

# Ordenar por clasificación y nombre
orden = {
    "Potencialmente previa": 1,
    "Disponibilidad por verificar": 2,
    "Posterior al accidente": 3
}

resultado["orden"] = resultado["clasificacion"].map(orden)

resultado = (
    resultado
    .sort_values(["orden", "variable"])
    .drop(columns="orden")
    .reset_index(drop=True)
)

# Resumen
print("\nRESUMEN DE CLASIFICACION")
print("-" * 75)

resumen = (
    resultado["clasificacion"]
    .value_counts()
)

for categoria, cantidad in resumen.items():
    print(f"{categoria}: {cantidad}")

print("\nMATRIZ DE DISPONIBILIDAD")
print("-" * 75)

print(resultado.to_string(index=False))

# Guardar CSV
resultado.to_csv(
    RUTA_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 75)
print("AUDITORIA TERMINADA")
print("=" * 75)
print("Variables revisadas:", len(resultado))
print("Archivo generado:", RUTA_SALIDA)
