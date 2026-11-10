"""Evalúa umbrales binarios para priorizar la clase fatal en V9D.

La selección se hace solo con 2018 usando un modelo entrenado hasta 2017.
Después se reentrena hasta 2018 y se informa una única vez sobre 2019-2020.
No reemplaza el modelo multiclasificador V9D existente.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from entrenar_modelo_v9d import (
    ARBOLES_FINALES,
    ARBOLES_VALIDACION,
    DATA_PATH,
    buscar_columna,
    crear_modelo,
    normalizar,
)


def puntuar(y_real, pred, clase_fatal):
    fatal = np.asarray(y_real) == clase_fatal
    pred_fatal = np.asarray(pred) == clase_fatal
    return {
        "accuracy": float(accuracy_score(y_real, pred)),
        "f1_macro": float(f1_score(y_real, pred, average="macro", zero_division=0)),
        "precision_fatal": float(precision_score(fatal, pred_fatal, zero_division=0)),
        "recall_fatal": float(recall_score(fatal, pred_fatal, zero_division=0)),
        "f1_fatal": float(f1_score(fatal, pred_fatal, zero_division=0)),
        "alertas_fatales": int(pred_fatal.sum()),
        "casos_fatales": int(fatal.sum()),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    target = buscar_columna(df, ["GRAVEDAD"])
    year = buscar_columna(df, ["ANO", "ANIO", "AÑO"])
    locality = buscar_columna(df, ["CODIGO_LOCALIDAD", "LOCALIDAD", "COD_LOCALIDAD"])
    numeric = [
        buscar_columna(df, ["ANO", "ANIO", "AÑO"]),
        buscar_columna(df, ["MES"]),
        buscar_columna(df, ["DIA", "DÍA"]),
        buscar_columna(df, ["DIA_SEMANA", "DÍA_SEMANA"]),
        buscar_columna(df, ["ES_FIN_DE_SEMANA"]),
        buscar_columna(df, ["HORA_NUM"]),
    ]
    if any(col is None for col in [target, year, locality, *numeric]):
        raise ValueError("Faltan columnas requeridas para V9D.")

    categorical = [locality]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[target] = df[target].astype("string").str.strip()
    df = df.dropna(subset=[year, target])
    df = df[df[target] != ""].copy()
    for col in categorical:
        df[col] = df[col].astype("string").fillna("DESCONOCIDO").astype(str)

    train_base = df[df[year] <= 2017]
    validation = df[df[year] == 2018]
    train_final = df[df[year] <= 2018]
    test = df[df[year].between(2019, 2020)]
    y_base = train_base[target].astype(str)
    y_val = validation[target].astype(str)
    labels_base = np.array(sorted(y_base.unique()))
    fatal_class = next((c for c in labels_base if "MUERTO" in normalizar(c)), None)
    if fatal_class is None:
        raise ValueError("No se encontró la clase fatal en entrenamiento.")

    base_weights = compute_class_weight("balanced", classes=labels_base, y=y_base)
    weights = dict(zip(labels_base, base_weights))
    weights[fatal_class] *= 2  # Factor seleccionado previamente solo con 2018.
    columns = numeric + categorical
    validation_model = crear_modelo(numeric, categorical, weights, ARBOLES_VALIDACION)
    validation_model.fit(train_base[columns], y_base)
    val_prob = validation_model.predict_proba(validation[columns])
    classes = validation_model.named_steps["clasificador"].classes_
    fatal_index = list(classes).index(fatal_class)
    val_fatal_prob = val_prob[:, fatal_index]

    val_is_fatal = (y_val.to_numpy() == fatal_class).astype(int)
    curve_precision, curve_recall, curve_thresholds = precision_recall_curve(
        val_is_fatal, val_fatal_prob
    )
    curve_f1 = (
        2 * curve_precision[:-1] * curve_recall[:-1]
        / np.maximum(curve_precision[:-1] + curve_recall[:-1], 1e-12)
    )
    validation_thresholds = {
        "max_f1_fatal": float(curve_thresholds[int(np.argmax(curve_f1))])
    }
    for target_recall in (0.50, 0.75):
        eligible = np.flatnonzero(curve_recall[:-1] >= target_recall)
        if len(eligible):
            best_index = eligible[np.argmax(curve_precision[eligible])]
            validation_thresholds[f"recall_{target_recall:.2f}"] = float(curve_thresholds[best_index])

    nonfatal_columns = [i for i in range(len(classes)) if i != fatal_index]
    nonfatal_val = classes[np.argmax(val_prob[:, nonfatal_columns], axis=1)]

    def score_validation_threshold(threshold):
        predictions = np.where(val_fatal_prob >= threshold, fatal_class, nonfatal_val)
        return {"umbral": threshold, **puntuar(y_val, predictions, fatal_class)}

    best = score_validation_threshold(validation_thresholds["max_f1_fatal"])
    recall_targets = {
        key: score_validation_threshold(value)
        for key, value in validation_thresholds.items()
        if key.startswith("recall_")
    }

    y_final = train_final[target].astype(str)
    labels_final = np.array(sorted(y_final.unique()))
    final_weights_arr = compute_class_weight("balanced", classes=labels_final, y=y_final)
    final_weights = dict(zip(labels_final, final_weights_arr))
    final_weights[fatal_class] *= 2
    final_model = crear_modelo(numeric, categorical, final_weights, ARBOLES_FINALES)
    final_model.fit(train_final[columns], y_final)
    probs = final_model.predict_proba(test[columns])
    final_classes = final_model.named_steps["clasificador"].classes_
    test_fatal_prob = probs[:, list(final_classes).index(fatal_class)]
    y_test = test[target].astype(str).to_numpy()
    argmax_pred = final_classes[np.argmax(probs, axis=1)]

    def threshold_prediction(threshold):
        nonfatal_cols = [i for i, c in enumerate(final_classes) if c != fatal_class]
        nonfatal = final_classes[np.argmax(probs[:, nonfatal_cols], axis=1)]
        return np.where(test_fatal_prob >= threshold, fatal_class, nonfatal)

    selected_pred = threshold_prediction(best["umbral"])
    val_argmax_pred = validation_model.predict(validation[columns])
    result = {
        "protocolo": "Selección del umbral en 2018; prueba temporal 2019-2020",
        "factor_peso_fatal": 2,
        "umbral_elegido_por_f1_fatal": best["umbral"],
        "validacion_argmax": puntuar(y_val, val_argmax_pred, fatal_class),
        "validacion_umbral_elegido": best,
        "validacion_umbral_para_recall": recall_targets,
        "prueba_argmax": puntuar(y_test, argmax_pred, fatal_class),
        "prueba_umbral_seleccionado_en_2018": puntuar(y_test, selected_pred, fatal_class),
        "nota": "El umbral se selecciona sin usar 2019-2020. El sistema sigue clasificando gravedad de siniestros registrados; no estima ocurrencia.",
    }
    out = Path(__file__).resolve().parents[2] / "models"
    (out / "evaluacion_umbral_muertos_v9d.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "GRAVEDAD_REAL": y_test,
        "PROBABILIDAD_FATAL": test_fatal_prob,
        "PREDICCION_ARGMAX": argmax_pred,
        "PREDICCION_UMBRAL_2018": selected_pred,
    }).to_csv(out / "predicciones_umbral_muertos_v9d.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
