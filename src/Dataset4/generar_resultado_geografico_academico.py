"""Prepara evidencia geográfica trazable, sin aprobar candidatos.

Cruza registros por dirección normalizada + código de localidad. El resultado
incluye las 50 filas con sus estados actuales y una capa GeoJSON aparte para
las 7 candidatas de cruce único pendientes de revisión humana.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ACCIDENTS = ROOT / "data/processed/siniestros_integrados.csv"
REVIEW = ROOT / "models/hoja_revision_manual_georreferenciacion.csv"
OUT = ROOT / "ENTREGA_ACADEMICA/geografia"


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip().upper()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(REVIEW, low_memory=False)
    events = pd.read_csv(
        ACCIDENTS,
        usecols=["CODIGO_ACCIDENTE", "FECHA", "HORA", "DIRECCION", "CODIGO_LOCALIDAD"],
        low_memory=False,
    )
    if len(geo) != 50:
        raise ValueError(f"Se esperaban las 50 candidatas; se encontraron {len(geo)}.")
    if not geo["VALIDADO"].astype(str).str.strip().str.upper().eq("NO").all():
        raise ValueError("Hay una candidata con VALIDADO distinto de NO. No se aprueba automáticamente.")

    geo["_direccion_key"] = geo["DIRECCION_NORMALIZADA"].map(normalize)
    geo["_localidad_key"] = pd.to_numeric(geo["CODIGO_LOCALIDAD"], errors="coerce").astype("Int64")
    events["_direccion_key"] = events["DIRECCION"].map(normalize)
    events["_localidad_key"] = pd.to_numeric(events["CODIGO_LOCALIDAD"], errors="coerce").astype("Int64")
    events["_fecha_parseada"] = pd.to_datetime(events["FECHA"], dayfirst=True, errors="coerce")
    counts = (events.dropna(subset=["_localidad_key"])
        .groupby(["_direccion_key", "_localidad_key"], dropna=False)
        .agg(SINIESTROS_ID_UNICOS=("CODIGO_ACCIDENTE", "nunique"),
             FECHA_PRIMER_REPORTE=("_fecha_parseada", "min"), FECHA_ULTIMO_REPORTE=("_fecha_parseada", "max"))
        .reset_index())
    result = geo.merge(
        counts,
        left_on=["_direccion_key", "_localidad_key"],
        right_on=["_direccion_key", "_localidad_key"],
        how="left",
        validate="many_to_one",
    )
    result["SINIESTROS_ID_UNICOS"] = result["SINIESTROS_ID_UNICOS"].fillna(0).astype(int)
    result["CORRESPONDENCIA"] = "COINCIDENCIA_TEXTUAL_DIRECCION_LOCALIDAD"
    result["ESTADO_VALIDACION_FINAL"] = "PENDIENTE_REVISION_HUMANA"
    result["ADVERTENCIA"] = (
        "La coincidencia textual no confirma ubicación física del evento; "
        "coordenada candidata no validada humanamente."
    )

    export_cols = [
        "ID_REVISION", "DIRECCION", "CODIGO_LOCALIDAD", "NOMBRE_LOCALIDAD",
        "GRUPO_ESPACIAL", "LONGITUD_CALCULADA", "LATITUD_CALCULADA",
        "ESTADO_REVISION", "VALIDADO", "PRIORIDAD_REVISION",
        "CANTIDAD_CRUCES", "TOTAL_GRUPOS_DIRECCION", "SINIESTROS_ID_UNICOS",
        "FECHA_PRIMER_REPORTE", "FECHA_ULTIMO_REPORTE", "CORRESPONDENCIA",
        "ESTADO_VALIDACION_FINAL", "ADVERTENCIA",
    ]
    result[export_cols].to_csv(OUT / "revision_50_candidatas_y_coincidencias.csv", index=False, encoding="utf-8-sig")

    unique_pending = result[result["ESTADO_REVISION"].eq("CANDIDATA_UNICA_PENDIENTE_VALIDACION")].copy()
    if len(unique_pending) != 7:
        raise ValueError(f"El estado esperado de 7 candidatas únicas cambió: {len(unique_pending)}.")
    unique_pending[export_cols].to_csv(OUT / "7_candidatas_unicas_pendientes.csv", index=False, encoding="utf-8-sig")

    features = []
    for row in unique_pending.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row.LONGITUD_CALCULADA), float(row.LATITUD_CALCULADA)]},
            "properties": {
                "id_revision": int(row.ID_REVISION),
                "direccion": str(row.DIRECCION),
                "localidad": str(row.NOMBRE_LOCALIDAD),
                "siniestros_con_coincidencia_textual": int(row.SINIESTROS_ID_UNICOS),
                "estado": str(row.ESTADO_REVISION),
                "validado": str(row.VALIDADO),
                "interpretacion": "Punto candidato. La coordenada y la correspondencia espacial requieren revisión humana.",
            },
        })
    (OUT / "7_candidatas_unicas_pendientes.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "name": "Candidatas unicas pendientes con coincidencia textual",
            "metadata": {
                "no_es_validacion_geografica": True,
                "metodo": "Coincidencia normalizada exacta de DIRECCION y CODIGO_LOCALIDAD",
                "registros_unicos_coincidentes": int(unique_pending["SINIESTROS_ID_UNICOS"].sum()),
                "crs": "EPSG:4326",
            },
            "features": features,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts_by_status = result.groupby("ESTADO_REVISION").size().to_dict()
    summary = {
        "candidatas_revisadas": int(len(result)),
        "validacion_humana_si": int(result["VALIDADO"].astype(str).str.upper().eq("SI").sum()),
        "estados": {str(key): int(value) for key, value in counts_by_status.items()},
        "registros_unicos_con_coincidencia_textual_en_7_candidatas_unicas_pendientes": int(unique_pending["SINIESTROS_ID_UNICOS"].sum()),
        "interpretacion": "Conteo por dirección y localidad exactas. No equivale a accidentes georreferenciados ni a coordenadas validadas.",
        "sumas_de_las_50_filas_no_aditivas": "Las filas ambiguas pueden repetir la misma combinación de dirección/localidad para ubicaciones geométricas alternativas.",
    }
    (OUT / "resumen_geografico_academico.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
