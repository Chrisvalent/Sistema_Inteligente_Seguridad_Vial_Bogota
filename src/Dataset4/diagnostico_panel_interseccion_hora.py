"""Diagnóstico conservador del panel intersección × fecha × hora.

Lee los registros integrados y el inventario de 50 ubicaciones candidatas.
No modifica datos de origen, modelos, API, base de datos ni credenciales.
Los cruces por texto son únicamente una medida de cobertura preliminar:
no constituyen geocodificación validada ni negativos verdaderos.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ACCIDENTS = ROOT / "data/processed/siniestros_integrados.csv"
GEOGRAPHY = ROOT / "models/hoja_revision_manual_georreferenciacion.csv"
OUT_MODELS = ROOT / "models"
OUT_POWERBI = ROOT / "ENTREGA_POWER_BI/06_EXPERIMENTO_PREVENTIVO"
YEARS = range(2015, 2021)


def normalize_text(values: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        value = unicodedata.normalize("NFKD", str(value or ""))
        value = "".join(c for c in value if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", value).strip().upper()

    return values.map(one)


def main() -> None:
    for source in (ACCIDENTS, GEOGRAPHY):
        if not source.exists():
            raise FileNotFoundError(source)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_POWERBI.mkdir(parents=True, exist_ok=True)

    accidents = pd.read_csv(ACCIDENTS, low_memory=False)
    geo = pd.read_csv(GEOGRAPHY, low_memory=False)
    required_accident = {"CODIGO_ACCIDENTE", "FECHA", "HORA", "DIRECCION", "CODIGO_LOCALIDAD"}
    required_geo = {"DIRECCION", "CODIGO_LOCALIDAD", "VALIDADO", "ESTADO_REVISION", "LONGITUD_CALCULADA", "LATITUD_CALCULADA"}
    if missing := required_accident.difference(accidents.columns):
        raise ValueError(f"Faltan campos del origen de siniestros: {sorted(missing)}")
    if missing := required_geo.difference(geo.columns):
        raise ValueError(f"Faltan campos del inventario geográfico: {sorted(missing)}")

    accidents["FECHA_PARSEADA"] = pd.to_datetime(accidents["FECHA"], dayfirst=True, errors="coerce")
    accidents["HORA_PARSEADA"] = pd.to_datetime(accidents["HORA"].astype(str), format="%H:%M:%S", errors="coerce")
    # Algunas fuentes guardan HH:MM. Completar el parseo sin inventar una hora.
    missing_time = accidents["HORA_PARSEADA"].isna()
    accidents.loc[missing_time, "HORA_PARSEADA"] = pd.to_datetime(
        accidents.loc[missing_time, "HORA"].astype(str), format="%H:%M", errors="coerce"
    )
    accidents["HORA_INICIO"] = accidents["HORA_PARSEADA"].dt.hour
    accidents["DIRECCION_KEY"] = normalize_text(accidents["DIRECCION"])
    geo["DIRECCION_KEY"] = normalize_text(geo["DIRECCION"])
    geo["ID_CANDIDATO_PANEL"] = range(1, len(geo) + 1)
    accidents["LOCALIDAD_KEY"] = accidents["CODIGO_LOCALIDAD"].astype("string").str.strip()
    geo["LOCALIDAD_KEY"] = geo["CODIGO_LOCALIDAD"].astype("string").str.strip()
    accidents["ANIO"] = accidents["FECHA_PARSEADA"].dt.year
    accidents = accidents[accidents["ANIO"].isin(YEARS)].copy()

    geo["COORDENADAS_PRESENTES"] = geo["LONGITUD_CALCULADA"].notna() & geo["LATITUD_CALCULADA"].notna()
    geo["VALIDADO_KEY"] = geo["VALIDADO"].astype(str).str.strip().str.upper()
    geo["ESTADO_KEY"] = geo["ESTADO_REVISION"].astype(str).str.strip().str.upper()

    # El join solo estima cuántos registros caen en el vocabulario de las
    # direcciones candidatas. Una dirección puede representar varias ubicaciones.
    keys = geo[["ID_CANDIDATO_PANEL", "DIRECCION_KEY", "LOCALIDAD_KEY"]].drop_duplicates()
    unique_location_keys = geo[["DIRECCION_KEY", "LOCALIDAD_KEY"]].drop_duplicates()
    matched = accidents.merge(keys.assign(COINCIDENCIA_CANDIDATA=1), how="left", on=["DIRECCION_KEY", "LOCALIDAD_KEY"])
    matched["COINCIDENCIA_CANDIDATA"] = matched["COINCIDENCIA_CANDIDATA"].fillna(0).astype(int)
    matched_candidate = matched[matched["COINCIDENCIA_CANDIDATA"].eq(1)].copy()
    matched_candidate["FECHA_KEY"] = matched_candidate["FECHA_PARSEADA"].dt.date.astype(str)
    hour_cells = matched_candidate.dropna(subset=["HORA_INICIO"]).drop_duplicates(
        ["ID_CANDIDATO_PANEL", "FECHA_KEY", "HORA_INICIO"]
    )

    annual_rows = []
    for year in YEARS:
        part = accidents[accidents["ANIO"].eq(year)]
        matched_year = matched_candidate[matched_candidate["ANIO"].eq(year)]
        cell_year = hour_cells[hour_cells["ANIO"].eq(year)]
        day_count = int(pd.Timestamp(year, 12, 31).dayofyear)
        theoretical_slots = int(len(geo) * day_count * 24)
        annual_rows.append({
            "ANIO": year,
            "SINIESTROS_REGISTRADOS": int(len(part)),
            "DIAS_CON_REGISTRO": int(part["FECHA_PARSEADA"].dt.date.nunique()),
            "DIRECCIONES_CANDIDATAS_CON_COINCIDENCIA": int(matched_year["DIRECCION_KEY"].nunique()),
            "SINIESTROS_EN_DIRECCION_CANDIDATA_TEXTO": int(matched_year["CODIGO_ACCIDENTE"].nunique()),
            "CELDAS_DIRECCION_FECHA_HORA_CON_REPORTE": int(len(cell_year)),
            "CELDAS_CANDIDATAS_HORA_TEORICAS": theoretical_slots,
            "CELDAS_SIN_REPORTE_REGISTRADO": max(0, theoretical_slots - int(len(cell_year))),
            "CEROS_CONFIRMADOS": 0,
            "HORAS_VALIDAS": int(part["HORA_INICIO"].notna().sum()),
        })
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(OUT_MODELS / "diagnostico_panel_anual.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT_POWERBI / "diagnostico_panel_anual.csv", index=False, encoding="utf-8-sig")

    split_rows = []
    for name, lo, hi in [("train", 2015, 2018), ("validation", 2019, 2019), ("test", 2020, 2020)]:
        p = accidents[accidents["ANIO"].between(lo, hi)]
        m = matched_candidate[matched_candidate["ANIO"].between(lo, hi)]
        split_rows.append({
            "SPLIT": name,
            "ANIOS": f"{lo}-{hi}" if lo != hi else str(lo),
            "SINIESTROS_REGISTRADOS": int(len(p)),
            "DIRECCIONES_CANDIDATAS_CON_COINCIDENCIA": int(m["DIRECCION_KEY"].nunique()),
            "SINIESTROS_EN_DIRECCION_CANDIDATA_TEXTO": int(m["CODIGO_ACCIDENTE"].nunique()),
            "APTO_PARA_EVALUACION": False,
            "MOTIVO": "Ubicaciones aún no validadas humanamente; ceros no confirmados como ausencia de siniestro.",
        })
    split = pd.DataFrame(split_rows)
    split.to_csv(OUT_MODELS / "diagnostico_panel_split_temporal.csv", index=False, encoding="utf-8-sig")
    split.to_csv(OUT_POWERBI / "diagnostico_panel_split_temporal.csv", index=False, encoding="utf-8-sig")

    geo_summary = (geo.groupby("ESTADO_KEY", dropna=False)
        .agg(CANDIDATOS=("DIRECCION", "size"), DIRECCIONES_DISTINTAS=("DIRECCION_KEY", "nunique"),
             CON_COORDENADAS=("COORDENADAS_PRESENTES", "sum"), VALIDACION_HUMANA=("VALIDADO_KEY", lambda x: int(x.eq("SI").sum())))
        .reset_index().rename(columns={"ESTADO_KEY": "ESTADO_REVISION"}))
    geo_summary.to_csv(OUT_MODELS / "diagnostico_panel_geografia.csv", index=False, encoding="utf-8-sig")
    geo_summary.to_csv(OUT_POWERBI / "diagnostico_panel_geografia.csv", index=False, encoding="utf-8-sig")

    hourly_rows = []
    days_in_period = sum(int(pd.Timestamp(year, 12, 31).dayofyear) for year in YEARS)
    for hour in range(24):
        in_hour = accidents[accidents["HORA_INICIO"].eq(hour)]
        in_candidates = matched_candidate[matched_candidate["HORA_INICIO"].eq(hour)]
        reported_candidate_cells = int(hour_cells[hour_cells["HORA_INICIO"].eq(hour)].shape[0])
        theoretical_hour_cells = int(len(geo) * days_in_period)
        hourly_rows.append({
            "HORA_INICIO": hour,
            "SINIESTROS_REGISTRADOS": int(len(in_hour)),
            "SINIESTROS_DIRECCION_CANDIDATA_TEXTO": int(in_candidates["CODIGO_ACCIDENTE"].nunique()),
            "CELDAS_DIRECCION_FECHA_HORA_CON_REPORTE": int(hour_cells[hour_cells["HORA_INICIO"].eq(hour)].shape[0]),
            "CELDAS_CANDIDATAS_HORA_TEORICAS": theoretical_hour_cells,
            "CELDAS_CANDIDATAS_HORA_CON_REPORTE": reported_candidate_cells,
            "CELDAS_SIN_REPORTE_REGISTRADO": max(0, theoretical_hour_cells - reported_candidate_cells),
            "CEROS_CONFIRMADOS": 0,
            "MOTIVO_CERO_NO_ESTIMABLE": "Las horas sin reporte son huecos provisionales; universo espacial y cobertura no están validados.",
        })
    hourly = pd.DataFrame(hourly_rows)
    hourly.to_csv(OUT_MODELS / "diagnostico_panel_horas.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(OUT_POWERBI / "diagnostico_panel_horas.csv", index=False, encoding="utf-8-sig")

    total = int(len(accidents))
    matches = int(matched_candidate["CODIGO_ACCIDENTE"].nunique())
    coord_count = int(geo["COORDENADAS_PRESENTES"].sum())
    valid_count = int(geo["VALIDADO_KEY"].eq("SI").sum())
    summary = {
        "diagnostico": "NO_VIABLE_POR_AHORA_PARA_ENTRENAR_O_VALIDAR_UN_PREDICTOR_PREVENTIVO",
        "unidad_objetivo": "intersección × fecha × hora",
        "periodo": {"inicio": "2015-01-01", "fin": "2020-12-31"},
        "fuente_siniestros": str(ACCIDENTS.relative_to(ROOT)),
        "fuente_geografica": str(GEOGRAPHY.relative_to(ROOT)),
        "siniestros": {
            "registros": total,
            "fechas_invalidas": int(pd.to_datetime(pd.read_csv(ACCIDENTS, usecols=["FECHA"])["FECHA"], dayfirst=True, errors="coerce").isna().sum()),
            "horas_validas": int(accidents["HORA_INICIO"].notna().sum()),
            "distribucion_anual": {str(r.ANIO): int(r.SINIESTROS_REGISTRADOS) for r in annual.itertuples()},
        },
        "geografia": {
            "filas_candidatas": int(len(geo)),
            "direcciones_textuales_distintas": int(geo["DIRECCION_KEY"].nunique()),
            "direcciones_localidad_distintas": int(unique_location_keys.shape[0]),
            "con_coordenadas_calculadas": coord_count,
            "validadas_humanamente": valid_count,
            "ambiguas": int(geo["ESTADO_KEY"].str.contains("AMBIGUA", na=False).sum()),
            "unicas_pendientes": int(geo["ESTADO_KEY"].str.contains("UNICA", na=False).sum()),
        },
        "cobertura_textual_no_validada": {
            "siniestros_coincidentes": matches,
            "porcentaje_sobre_todos_los_siniestros": round(matches / total * 100, 3) if total else None,
            "advertencia": "La coincidencia de dirección no confirma que el accidente ocurrió en la intersección ni asigna una coordenada correcta.",
        },
        "casos": {
            "positivos": "Una o más ocurrencias registradas en una intersección/hora, solo después de validar ubicación y deduplicar el evento.",
            "negativos": "No construir aún. La ausencia de una fila puede ser cero de reporte, falta de cobertura o ausencia de exposición; no es un negativo verdadero.",
            "panel_completo": "No construir mientras el universo espacial no esté consolidado y las horas sin reporte no puedan interpretarse.",
        },
        "split_temporal": {"train": "2015-2018", "validation": "2019", "test": "2020", "estado": "definido, no ejecutado por falta de etiquetas/universo válidos"},
        "variables": {
            "candidatas_temporales": ["año", "mes", "día de semana", "hora", "festivo si se incorpora calendario externo fechado"],
            "candidatas_espaciales": ["ID de intersección", "coordenadas", "localidad"],
            "condicion": "ID/coordenadas/localidad solo tras curación y validación espacial; fecha y hora describen el periodo del resultado y deben poder conocerse al momento de emitir la predicción.",
            "no_disponibles_en_las_fuentes_revisadas": ["flujo/aforo vehicular", "exposición por intersección y hora", "clima histórico alineado", "inventario temporal de infraestructura/señalización"],
            "excluir": ["gravedad", "clase del accidente", "hipótesis/causas", "actores y vehículos", "variables derivadas tras el siniestro"],
        },
        "metricas": "PR-AUC, Brier y recall@K solo después de confirmar etiquetas positivas/negativas y cobertura; no calculadas.",
        "modelos_existentes": "V8B y V9D no se evalúan ni presentan como predictores preventivos de ocurrencia.",
        "salidas": [
            "models/diagnostico_panel_anual.csv",
            "models/diagnostico_panel_split_temporal.csv",
            "models/diagnostico_panel_geografia.csv",
            "models/diagnostico_panel_horas.csv",
            "ENTREGA_POWER_BI/06_EXPERIMENTO_PREVENTIVO/",
        ],
    }
    for path in (OUT_MODELS / "diagnostico_panel_interseccion_hora.json", OUT_POWERBI / "diagnostico_panel_interseccion_hora.json"):
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "diagnostico": summary["diagnostico"],
        "siniestros": total,
        "candidatos": len(geo),
        "con_coordenadas_calculadas": coord_count,
        "validados_humanamente": valid_count,
        "ambiguos": summary["geografia"]["ambiguas"],
        "unicos_pendientes": summary["geografia"]["unicas_pendientes"],
        "coincidencias_textuales": matches,
        "salidas": summary["salidas"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
