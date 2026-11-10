"""Prepara una cola trazable para revisión manual de cruces candidatos."""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


BASE = Path(__file__).resolve().parents[2]
ARCHIVO_CALIDAD = BASE / "models" / "revision_calidad_georreferenciacion.csv"
ARCHIVO_RESUMEN = BASE / "models" / "resumen_cruces_normalizados.csv"
ARCHIVO_LOCALIDADES = BASE / "data" / "raw" / "poligonos-localidad-2.geojson"
ARCHIVO_SALIDA = BASE / "models" / "hoja_revision_manual_georreferenciacion.csv"


def clasificar_revision(fila):
    """Clasifica ambigüedad; nunca aprueba coordenadas automáticamente."""
    if pd.isna(fila.get("LONGITUD_CALCULADA")) or pd.isna(
        fila.get("LATITUD_CALCULADA")
    ):
        return "SIN_CANDIDATA_GEOMETRICA"

    cantidad_cruces = pd.to_numeric(
        fila.get("CANTIDAD_CRUCES"), errors="coerce"
    )
    grupos_direccion = pd.to_numeric(
        fila.get("TOTAL_GRUPOS_DIRECCION"), errors="coerce"
    )

    if pd.isna(cantidad_cruces) or pd.isna(grupos_direccion):
        return "DATOS_INCOMPLETOS_REQUIERE_REVISION"

    cantidad_cruces = int(cantidad_cruces)
    grupos_direccion = int(grupos_direccion)

    if cantidad_cruces < 1 or grupos_direccion < 1:
        return "SIN_CANDIDATA_GEOMETRICA"

    if cantidad_cruces > 1 or grupos_direccion > 1:
        return "AMBIGUA_REQUIERE_REVISION"

    return "CANDIDATA_UNICA_PENDIENTE_VALIDACION"


def coordenada_en_localidad(fila, geometria_por_codigo):
    """Comprueba pertenencia al polígono; no certifica la intersección."""
    longitud = pd.to_numeric(fila.get("LONGITUD_CALCULADA"), errors="coerce")
    latitud = pd.to_numeric(fila.get("LATITUD_CALCULADA"), errors="coerce")
    codigo = pd.to_numeric(fila.get("CODIGO_LOCALIDAD"), errors="coerce")
    if pd.isna(longitud) or pd.isna(latitud) or pd.isna(codigo):
        return "NO_DISPONIBLE"

    poligono = geometria_por_codigo.get(str(int(codigo)).zfill(2))
    if poligono is None:
        return "NO_DISPONIBLE"

    return "SI" if poligono.covers(Point(longitud, latitud)) else "NO"


def construir_hoja(calidad, resumen, geometria_por_codigo=None):
    requeridas_calidad = {
        "DIRECCION_NORMALIZADA",
        "CODIGO_LOCALIDAD",
        "GRUPO_ESPACIAL",
        "LONGITUD_CALCULADA",
        "LATITUD_CALCULADA",
        "CANTIDAD_CRUCES",
        "TOTAL_GRUPOS_DIRECCION",
        "VALIDADO",
    }
    requeridas_resumen = {
        "DIRECCION_NORMALIZADA",
        "CODIGO_LOCALIDAD",
        "CANTIDAD_REGISTROS_CRUCE",
    }

    faltan_calidad = requeridas_calidad - set(calidad.columns)
    faltan_resumen = requeridas_resumen - set(resumen.columns)
    if faltan_calidad or faltan_resumen:
        raise ValueError(
            f"Columnas faltantes en calidad: {sorted(faltan_calidad)}; "
            f"en resumen: {sorted(faltan_resumen)}"
        )

    claves = ["CODIGO_LOCALIDAD", "DIRECCION_NORMALIZADA"]
    if resumen.duplicated(claves).any():
        raise ValueError("El resumen contiene claves normalizadas duplicadas.")

    hoja = calidad.merge(
        resumen[claves + ["CANTIDAD_REGISTROS_CRUCE"]],
        on=claves,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    sin_resumen = int((hoja["_merge"] != "both").sum())
    if sin_resumen:
        raise ValueError(
            f"{sin_resumen} candidatos no tienen resumen de registros asociados."
        )
    hoja = hoja.drop(columns="_merge")

    hoja["ESTADO_REVISION"] = hoja.apply(clasificar_revision, axis=1)
    if geometria_por_codigo is not None:
        hoja["COORDENADA_EN_LOCALIDAD_DECLARADA"] = hoja.apply(
            coordenada_en_localidad,
            axis=1,
            geometria_por_codigo=geometria_por_codigo,
        )
        fuera = hoja["COORDENADA_EN_LOCALIDAD_DECLARADA"].eq("NO")
        hoja.loc[fuera, "ESTADO_REVISION"] = (
            "COORDENADA_FUERA_LOCALIDAD_REQUIERE_REVISION"
        )

    hoja["PRIORIDAD_REVISION"] = hoja["ESTADO_REVISION"].map(
        {
            "AMBIGUA_REQUIERE_REVISION": "ALTA",
            "SIN_CANDIDATA_GEOMETRICA": "ALTA",
            "DATOS_INCOMPLETOS_REQUIERE_REVISION": "ALTA",
            "COORDENADA_FUERA_LOCALIDAD_REQUIERE_REVISION": "ALTA",
            "CANDIDATA_UNICA_PENDIENTE_VALIDACION": "MEDIA",
        }
    )

    # Campos para que el revisor documente evidencia sin reemplazar
    # las coordenadas calculadas ni aprobar automáticamente un candidato.
    for columna in (
        "LONGITUD_APROBADA",
        "LATITUD_APROBADA",
        "FUENTE_VERIFICACION",
        "REVISOR",
        "FECHA_REVISION",
        "OBSERVACIONES_REVISION",
    ):
        hoja[columna] = ""

    if not hoja["VALIDADO"].astype(str).str.upper().eq("NO").all():
        raise ValueError(
            "La auditoría de origen contiene candidatos marcados como validados; "
            "revísalos antes de generar la cola."
        )

    hoja = hoja.sort_values(
        ["PRIORIDAD_REVISION", "CANTIDAD_REGISTROS_CRUCE", "DIRECCION_NORMALIZADA"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    hoja.insert(0, "ID_REVISION", range(1, len(hoja) + 1))
    return hoja


def main():
    if (
        not ARCHIVO_CALIDAD.exists()
        or not ARCHIVO_RESUMEN.exists()
        or not ARCHIVO_LOCALIDADES.exists()
    ):
        raise FileNotFoundError(
            "Faltan las auditorías, el resumen o los polígonos de localidades."
        )

    calidad = pd.read_csv(ARCHIVO_CALIDAD, low_memory=False)
    resumen = pd.read_csv(ARCHIVO_RESUMEN, low_memory=False)
    localidades = gpd.read_file(ARCHIVO_LOCALIDADES)
    if localidades.crs is None or str(localidades.crs).upper() != "EPSG:4326":
        localidades = localidades.to_crs("EPSG:4326")
    geometria_por_codigo = {
        str(int(row["identificador_unico_de_la_localidad"])).zfill(2): row.geometry
        for _, row in localidades.iterrows()
    }
    hoja = construir_hoja(calidad, resumen, geometria_por_codigo)
    ARCHIVO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    hoja.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

    print(f"Candidatos incluidos: {len(hoja)}")
    print("Estados de revisión:")
    print(hoja["ESTADO_REVISION"].value_counts().to_string())
    print(f"VALIDADO=NO: {int(hoja['VALIDADO'].astype(str).str.upper().eq('NO').sum())}")
    print(
        "Dentro de localidad declarada:",
        int(hoja["COORDENADA_EN_LOCALIDAD_DECLARADA"].eq("SI").sum()),
    )
    print(
        "Fuera de localidad declarada:",
        int(hoja["COORDENADA_EN_LOCALIDAD_DECLARADA"].eq("NO").sum()),
    )
    print(f"Hoja creada: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
