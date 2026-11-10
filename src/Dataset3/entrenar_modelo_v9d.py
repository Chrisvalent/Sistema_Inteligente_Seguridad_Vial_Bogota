
"""
V9D - Modelo preventivo con ajuste de pesos por clase.

Entrenamiento para selección: años hasta 2017.
Validación: año 2018.
Entrenamiento final: años hasta 2018.
Prueba final: años 2019-2020.

Variables preventivas:
- Año, mes, día, día de la semana
- Fin de semana y hora
- Localidad
- Diseño de lugar, si está disponible y representa
  una característica vial conocida antes del accidente.

No utiliza CLASE, CHOQUE, actores, vehículos, hipótesis
ni causas posteriores al siniestro.
"""

from pathlib import Path
import json
import unicodedata
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "siniestros_modelo.csv"
)

# Solo activar después de confirmar que DISENO_LUGAR representa una
# característica estable de la vía disponible antes del accidente.
INCLUIR_DISENO_LUGAR = False

# Factores candidatos para aumentar el peso de "Con Muertos".
FACTORES_MUERTOS = [1, 2, 3, 5, 8]

ARBOLES_VALIDACION = 80
ARBOLES_FINALES = 150
SEMILLA = 42


def buscar_columna(df, opciones):
    columnas = {str(c).strip().upper(): c for c in df.columns}
    for opcion in opciones:
        if opcion.upper() in columnas:
            return columnas[opcion.upper()]
    return None


def normalizar(texto):
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(
        c for c in texto if not unicodedata.combining(c)
    )


def crear_modelo(numericas, categoricas, peso_clases, arboles):
    transformadores = [
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas,
        )
    ]

    if categoricas:
        transformadores.append(
            (
                "categoricas",
                Pipeline([
                    (
                        "imputador",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categoricas,
            )
        )

    preprocesamiento = ColumnTransformer(transformadores)

    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        (
            "clasificador",
            RandomForestClassifier(
                n_estimators=arboles,
                max_features="sqrt",
                min_samples_leaf=5,
                class_weight=peso_clases,
                random_state=SEMILLA,
                n_jobs=-1,
                verbose=0,
            ),
        ),
    ])


def evaluar_muertos(y_real, y_pred, etiqueta_muertos):
    precision = precision_score(
        y_real,
        y_pred,
        labels=[etiqueta_muertos],
        average="micro",
        zero_division=0,
    )
    recall = recall_score(
        y_real,
        y_pred,
        labels=[etiqueta_muertos],
        average="micro",
        zero_division=0,
    )
    f1 = f1_score(
        y_real,
        y_pred,
        labels=[etiqueta_muertos],
        average="micro",
        zero_division=0,
    )

    return float(precision), float(recall), float(f1)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    objetivo = buscar_columna(df, ["GRAVEDAD"])
    anio = buscar_columna(df, ["ANO", "ANIO", "AÑO"])
    localidad = buscar_columna(
        df, ["CODIGO_LOCALIDAD", "LOCALIDAD", "COD_LOCALIDAD"]
    )

    mapa = {
        "MES": buscar_columna(df, ["MES"]),
        "DIA": buscar_columna(df, ["DIA", "DÍA"]),
        "DIA_SEMANA": buscar_columna(
            df, ["DIA_SEMANA", "DÍA_SEMANA"]
        ),
        "ES_FIN_DE_SEMANA": buscar_columna(
            df, ["ES_FIN_DE_SEMANA"]
        ),
        "HORA_NUM": buscar_columna(df, ["HORA_NUM"]),
    }

    requeridas = {
        "GRAVEDAD": objetivo,
        "ANO": anio,
        "LOCALIDAD": localidad,
        **mapa,
    }

    faltantes = [k for k, v in requeridas.items() if v is None]
    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}\n"
            f"Disponibles: {list(df.columns)}"
        )

    diseno = buscar_columna(df, ["DISENO_LUGAR", "DISEÑO_LUGAR"])

    usar_diseno = INCLUIR_DISENO_LUGAR and diseno is not None
    sufijo_salida = "con_diseno" if usar_diseno else "sin_diseno"

    if INCLUIR_DISENO_LUGAR and diseno is None:
        print("Aviso: DISENO_LUGAR no existe; se omitirá.")

    numericas = [
        anio,
        mapa["MES"],
        mapa["DIA"],
        mapa["DIA_SEMANA"],
        mapa["ES_FIN_DE_SEMANA"],
        mapa["HORA_NUM"],
    ]

    categoricas = [localidad]
    if usar_diseno:
        categoricas.append(diseno)

    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[objetivo] = df[objetivo].astype("string").str.strip()
    df = df.dropna(subset=[anio, objetivo]).copy()
    df = df[df[objetivo] != ""].copy()

    # Asegurar valores categóricos consistentes.
    for col in categoricas:
        df[col] = (
            df[col].astype("string")
            .fillna("DESCONOCIDO")
            .astype(str)
        )

    # Divisiones temporales.
    train_base = df[df[anio] <= 2017].copy()
    validacion = df[df[anio] == 2018].copy()
    train_final = df[df[anio] <= 2018].copy()
    test = df[(df[anio] >= 2019) & (df[anio] <= 2020)].copy()

    if any(x.empty for x in [train_base, validacion, train_final, test]):
        raise ValueError(
            "Alguna partición temporal quedó vacía. "
            f"Años disponibles: {sorted(df[anio].dropna().unique())}"
        )

    y_base = train_base[objetivo].astype(str)
    y_val = validacion[objetivo].astype(str)

    etiquetas_base = np.array(sorted(y_base.unique()))

    etiqueta_muertos = next(
        (
            etiqueta for etiqueta in etiquetas_base
            if "MUERTO" in normalizar(etiqueta)
        ),
        None,
    )

    if etiqueta_muertos is None:
        raise ValueError(
            "No se encontró la clase 'Con Muertos' "
            f"en entrenamiento. Clases: {etiquetas_base.tolist()}"
        )

    pesos_base = compute_class_weight(
        class_weight="balanced",
        classes=etiquetas_base,
        y=y_base,
    )
    pesos_base = dict(zip(etiquetas_base, pesos_base))

    X_base = train_base[numericas + categoricas]
    X_val = validacion[numericas + categoricas]

    print("=" * 68)
    print("MODELO V9D - CLASIFICACIÓN TEMPORAL DE GRAVEDAD")
    print("=" * 68)
    print(f"Entrenamiento de selección (hasta 2017): {len(train_base):,}")
    print(f"Validación (2018): {len(validacion):,}")
    print(f"Entrenamiento final (hasta 2018): {len(train_final):,}")
    print(f"Prueba final (2019-2020): {len(test):,}")
    print("Variables:", numericas + categoricas)
    print("Clase minoritaria objetivo:", etiqueta_muertos)

    # Comparar pesos usando solo la validación de 2018.
    resultados_validacion = []
    mejor_factor = None
    mejor_f1 = -1.0

    print("\nComparando factores de peso en validación...")

    for factor in FACTORES_MUERTOS:
        pesos = pesos_base.copy()
        pesos[etiqueta_muertos] *= factor

        modelo_val = crear_modelo(
            numericas,
            categoricas,
            pesos,
            ARBOLES_VALIDACION,
        )

        print(f"\nProbando factor {factor}...")
        modelo_val.fit(X_base, y_base)

        pred_val = modelo_val.predict(X_val)

        precision, recall, f1 = evaluar_muertos(
            y_val, pred_val, etiqueta_muertos
        )

        resultados_validacion.append({
            "factor_muertos": factor,
            "precision_muertos": precision,
            "recall_muertos": recall,
            "f1_muertos": f1,
        })

        print(
            f"Factor={factor} | "
            f"Precision muertos={precision:.4f} | "
            f"Recall muertos={recall:.4f} | "
            f"F1 muertos={f1:.4f}"
        )

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_factor = factor

    print("\nFactor elegido según F1 de 'Con Muertos':", mejor_factor)

    # Reentrenar con todos los datos hasta 2018.
    pesos_finales = pesos_base.copy()
    pesos_finales[etiqueta_muertos] *= mejor_factor

    # Recalcular pesos base sobre el conjunto final.
    y_train_final = train_final[objetivo].astype(str)
    etiquetas_finales = np.array(sorted(y_train_final.unique()))
    pesos_finales_base = compute_class_weight(
        class_weight="balanced",
        classes=etiquetas_finales,
        y=y_train_final,
    )
    pesos_finales = dict(zip(etiquetas_finales, pesos_finales_base))
    pesos_finales[etiqueta_muertos] *= mejor_factor

    modelo = crear_modelo(
        numericas,
        categoricas,
        pesos_finales,
        ARBOLES_FINALES,
    )

    X_train_final = train_final[numericas + categoricas]
    X_test = test[numericas + categoricas]
    y_test = test[objetivo].astype(str)

    print("\nEntrenando modelo final con datos hasta 2018...")
    modelo.fit(X_train_final, y_train_final)
    pred_test = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, pred_test)
    balanced = balanced_accuracy_score(y_test, pred_test)
    f1_macro = f1_score(
        y_test, pred_test, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        y_test, pred_test, average="weighted", zero_division=0
    )

    precision, recall, f1_muertos = evaluar_muertos(
        y_test, pred_test, etiqueta_muertos
    )

    etiquetas = sorted(
        set(y_train_final.unique()) | set(y_test.unique())
    )
    matriz = confusion_matrix(y_test, pred_test, labels=etiquetas)

    print("\n" + "=" * 68)
    print("RESULTADOS FINALES V9D - PRUEBA 2019-2020")
    print("=" * 68)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"F1 macro:          {f1_macro:.4f}")
    print(f"F1 weighted:       {f1_weighted:.4f}")
    print(f"Precision muertos: {precision:.4f}")
    print(f"Recall muertos:    {recall:.4f}")
    print(f"F1 muertos:        {f1_muertos:.4f}")

    print("\nReporte por clase:")
    print(classification_report(
        y_test, pred_test, labels=etiquetas, zero_division=0
    ))
    print("Orden de clases:", etiquetas)
    print("Matriz de confusión:")
    print(matriz)

    carpeta = PROJECT_ROOT / "models"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_modelo = carpeta / f"modelo_v9d_{sufijo_salida}.pkl"
    ruta_metricas = carpeta / f"metricas_modelo_v9d_{sufijo_salida}.json"
    ruta_predicciones = carpeta / f"predicciones_v9d_{sufijo_salida}.csv"

    joblib.dump(modelo, ruta_modelo)

    metricas = {
        "version": f"V9D_clasificacion_gravedad_temporal_{sufijo_salida}",
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "entrenamiento_seleccion_hasta": 2017,
        "validacion_anio": 2018,
        "entrenamiento_final_hasta": 2018,
        "prueba_desde": 2019,
        "prueba_hasta": 2020,
        "variables": [str(c) for c in numericas + categoricas],
        "incluye_diseno_lugar": bool(usar_diseno),
        "factores_muertos_probados": FACTORES_MUERTOS,
        "resultados_validacion": resultados_validacion,
        "factor_muertos_elegido": int(mejor_factor),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_muertos": precision,
        "recall_muertos": recall,
        "f1_muertos": f1_muertos,
        "orden_matriz_confusion": etiquetas,
        "matriz_confusion": matriz.tolist(),
        "reporte_por_clase": classification_report(
            y_test,
            pred_test,
            labels=etiquetas,
            output_dict=True,
            zero_division=0,
        ),
        "nota": (
            "Predice gravedad entre siniestros registrados; "
            "no estima por sí solo la probabilidad de ocurrencia. "
            "El factor se seleccionó usando únicamente 2018."
        ),
    }

    ruta_metricas.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({
        "GRAVEDAD_REAL": y_test.to_numpy(),
        "GRAVEDAD_PREDICHA": pred_test,
    }).to_csv(
        ruta_predicciones,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nArchivos generados:")
    print(ruta_modelo)
    print(ruta_metricas)
    print(ruta_predicciones)


if __name__ == "__main__":
    main()
