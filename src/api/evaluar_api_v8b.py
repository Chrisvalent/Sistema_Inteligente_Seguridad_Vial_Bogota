
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

RUTA_CSV = (
    RAIZ / "data" / "processed" / "siniestros_modelo.csv"
)

RUTA_MODELO = RAIZ / "models" / "modelo_v8b.pkl"

CARPETA_SALIDA = RAIZ / "models"

N_MUESTRA = 2000
SEMILLA = 42

UMBRAL_MUERTOS = 0.30
CLASE_FATAL = "Con Muertos"

ARCHIVO_PREDICCIONES = (
    CARPETA_SALIDA / "evaluacion_api_v8b_predicciones.csv"
)

ARCHIVO_METRICAS = (
    CARPETA_SALIDA / "evaluacion_api_v8b_metricas.json"
)

ARCHIVO_MATRIZ = (
    CARPETA_SALIDA / "evaluacion_api_v8b_matriz_confusion.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def clasificar_con_umbral(probabilidades, clases):
    """
    Reproduce la lógica de predicción del endpoint de la API.

    Si la puntuación de Con Muertos alcanza el umbral, predice
    Con Muertos. En caso contrario, selecciona la clase no fatal
    con mayor puntuación.
    """
    mapa = {
        str(clase): float(probabilidad)
        for clase, probabilidad in zip(clases, probabilidades)
    }

    puntuacion_fatal = mapa[CLASE_FATAL]

    if puntuacion_fatal >= UMBRAL_MUERTOS:
        return CLASE_FATAL

    clases_no_fatales = [
        clase for clase in mapa
        if clase != CLASE_FATAL
    ]

    return max(
        clases_no_fatales,
        key=lambda clase: mapa[clase],
    )


def validar_archivos():
    if not RUTA_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV:\n{RUTA_CSV}"
        )

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{RUTA_MODELO}"
        )

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():
    print("=" * 70)
    print("EVALUACIÓN ADICIONAL DEL MODELO V8B")
    print("=" * 70)

    validar_archivos()

    # --------------------------------------------------------
    # 1. CARGAR MODELO
    # --------------------------------------------------------

    print("\nCargando modelo...")
    modelo = joblib.load(RUTA_MODELO)

    if not hasattr(modelo, "feature_names_in_"):
        raise ValueError(
            "El modelo no expone feature_names_in_; "
            "no se pueden verificar las columnas automáticamente."
        )

    variables = list(modelo.feature_names_in_)
    clases = [str(clase) for clase in modelo.classes_]

    if CLASE_FATAL not in clases:
        raise ValueError(
            f"La clase {CLASE_FATAL!r} no existe en el modelo. "
            f"Clases encontradas: {clases}"
        )

    # --------------------------------------------------------
    # 2. CARGAR Y VALIDAR DATOS
    # --------------------------------------------------------

    print("Cargando datos...")
    df = pd.read_csv(RUTA_CSV)

    if "GRAVEDAD" not in df.columns:
        raise ValueError(
            "El CSV no contiene la columna objetivo GRAVEDAD."
        )

    faltantes = [
        columna for columna in variables
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan variables requeridas por el modelo: {faltantes}"
        )

    # Excluir registros incompletos para esta prueba.
    datos = df.dropna(
        subset=variables + ["GRAVEDAD"]
    ).copy()

    # Evaluar solo las clases conocidas por el modelo.
    datos["GRAVEDAD"] = datos["GRAVEDAD"].astype(str)

    datos = datos[
        datos["GRAVEDAD"].isin(clases)
    ].copy()

    if datos.empty:
        raise ValueError(
            "No existen registros completos con clases reconocidas."
        )

    # --------------------------------------------------------
    # 3. SELECCIONAR MUESTRA
    # --------------------------------------------------------

    n = min(N_MUESTRA, len(datos))

    muestra = datos.sample(
        n=n,
        random_state=SEMILLA,
    ).copy()

    X = muestra[variables]
    y_real = muestra["GRAVEDAD"].to_numpy()

    print(f"Registros completos disponibles: {len(datos):,}")
    print(f"Registros evaluados: {len(muestra):,}")
    print(f"Variables utilizadas: {len(variables)}")

    # --------------------------------------------------------
    # 4. OBTENER PUNTUACIONES Y PREDICCIONES
    # --------------------------------------------------------

    print("Calculando puntuaciones...")

    probabilidades = modelo.predict_proba(X)

    indice_fatal = clases.index(CLASE_FATAL)

    puntuacion_fatal = probabilidades[:, indice_fatal]

    y_pred = np.array([
        clasificar_con_umbral(fila_prob, clases)
        for fila_prob in probabilidades
    ])

    # --------------------------------------------------------
    # 5. MÉTRICAS GENERALES
    # --------------------------------------------------------

    accuracy = accuracy_score(y_real, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_real, y_pred
    )

    f1_macro = f1_score(
        y_real,
        y_pred,
        labels=clases,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_real,
        y_pred,
        labels=clases,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # 6. MÉTRICAS DE CON MUERTOS
    # --------------------------------------------------------

    # Convertimos únicamente para evaluar esta clase:
    # 1 = Con Muertos; 0 = cualquier otra clase.
    y_real_fatal = (
        y_real == CLASE_FATAL
    ).astype(int)

    y_pred_fatal = (
        y_pred == CLASE_FATAL
    ).astype(int)

    precision_fatal = precision_score(
        y_real_fatal,
        y_pred_fatal,
        zero_division=0,
    )

    recall_fatal = recall_score(
        y_real_fatal,
        y_pred_fatal,
        zero_division=0,
    )

    f1_fatal = f1_score(
        y_real_fatal,
        y_pred_fatal,
        zero_division=0,
    )

    verdaderos_fatales = int(
        np.sum((y_real_fatal == 1) & (y_pred_fatal == 1))
    )

    falsos_positivos_fatales = int(
        np.sum((y_real_fatal == 0) & (y_pred_fatal == 1))
    )

    falsos_negativos_fatales = int(
        np.sum((y_real_fatal == 1) & (y_pred_fatal == 0))
    )

    # --------------------------------------------------------
    # 7. REPORTE Y MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    reporte_texto = classification_report(
        y_real,
        y_pred,
        labels=clases,
        zero_division=0,
    )

    reporte_json = classification_report(
        y_real,
        y_pred,
        labels=clases,
        output_dict=True,
        zero_division=0,
    )

    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=clases,
    )

    df_matriz = pd.DataFrame(
        matriz,
        index=[f"Real: {clase}" for clase in clases],
        columns=[f"Predicho: {clase}" for clase in clases],
    )

    # --------------------------------------------------------
    # 8. GUARDAR PREDICCIONES
    # --------------------------------------------------------

    columnas_resultado = [
        columna for columna in [
            "ANO",
            "MES",
            "DIA",
            "CODIGO_LOCALIDAD",
            "GRAVEDAD",
        ]
        if columna in muestra.columns
    ]

    resultados = muestra[columnas_resultado].copy()

    resultados = resultados.rename(
        columns={"GRAVEDAD": "gravedad_real"}
    )

    resultados["prediccion"] = y_pred

    resultados["puntuacion_modelo_con_muertos"] = (
        puntuacion_fatal
    )

    resultados["supera_umbral_con_muertos"] = (
        puntuacion_fatal >= UMBRAL_MUERTOS
    )

    resultados.to_csv(
        ARCHIVO_PREDICCIONES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 9. GUARDAR MATRIZ
    # --------------------------------------------------------

    df_matriz.to_csv(
        ARCHIVO_MATRIZ,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 10. GUARDAR MÉTRICAS JSON
    # --------------------------------------------------------

    metricas = {
        "modelo": "Random Forest V8B",
        "tipo_evaluacion": (
            "Muestra aleatoria de registros completos del CSV procesado. "
            "Es una verificación adicional, no una evaluación independiente."
        ),
        "semilla": SEMILLA,
        "registros_disponibles": int(len(datos)),
        "registros_evaluados": int(len(muestra)),
        "numero_variables": len(variables),
        "umbral_con_muertos": UMBRAL_MUERTOS,
        "clases": clases,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_con_muertos": float(precision_fatal),
        "recall_con_muertos": float(recall_fatal),
        "f1_con_muertos": float(f1_fatal),
        "verdaderos_positivos_con_muertos": verdaderos_fatales,
        "falsos_positivos_con_muertos": falsos_positivos_fatales,
        "falsos_negativos_con_muertos": falsos_negativos_fatales,
        "reporte_por_clase": reporte_json,
        "matriz_confusion": matriz.tolist(),
        "nota": (
            "Las puntuaciones del modelo no deben interpretarse como "
            "probabilidades calibradas. El conjunto contiene variables "
            "que pueden registrarse después del accidente. Esta prueba "
            "no valida el uso preventivo del modelo."
        ),
    }

    with open(
        ARCHIVO_METRICAS,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            metricas,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 11. MOSTRAR RESULTADOS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTADOS DE LA MUESTRA")
    print("=" * 70)

    print(f"Accuracy:                  {accuracy:.4f}")
    print(f"Balanced accuracy:         {balanced_accuracy:.4f}")
    print(f"F1 macro:                  {f1_macro:.4f}")
    print(f"F1 ponderado:              {f1_weighted:.4f}")
    print(f"Precisión Con Muertos:     {precision_fatal:.4f}")
    print(f"Recall Con Muertos:        {recall_fatal:.4f}")
    print(f"F1 Con Muertos:            {f1_fatal:.4f}")

    print("\nErrores para Con Muertos:")
    print(f"Verdaderos positivos:      {verdaderos_fatales}")
    print(f"Falsos positivos:          {falsos_positivos_fatales}")
    print(f"Falsos negativos:          {falsos_negativos_fatales}")

    print("\nReporte por clase:")
    print(reporte_texto)

    print("Matriz de confusión:")
    print(df_matriz.to_string())

    print("\nArchivos generados:")
    print(f"- {ARCHIVO_METRICAS}")
    print(f"- {ARCHIVO_MATRIZ}")
    print(f"- {ARCHIVO_PREDICCIONES}")

    print("\nEvaluación finalizada correctamente.")


if __name__ == "__main__":
    main()
