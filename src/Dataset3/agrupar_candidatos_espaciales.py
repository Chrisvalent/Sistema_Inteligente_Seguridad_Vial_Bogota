
from pathlib import Path

import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN


# ============================================================
# RUTAS
# ============================================================

BASE = Path(__file__).resolve().parents[2]

ENTRADA = (
    BASE / "models" / "candidatos_filtrados_localidad.csv"
)

SALIDA = (
    BASE / "models" / "candidatos_agrupados_espacialmente.csv"
)

RESUMEN = (
    BASE / "models" / "resumen_agrupacion_espacial.csv"
)

DISTANCIA_METROS = 20


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("AGRUPACIÓN ESPACIAL DE CANDIDATOS")
    print("=" * 70)

    if not ENTRADA.exists():
        raise FileNotFoundError(ENTRADA)

    df = pd.read_csv(
        ENTRADA,
        encoding="utf-8-sig"
    )

    df = df[
        df["DENTRO_LOCALIDAD"].astype(str).str.lower()
        .isin(["true", "1"])
    ].copy()

    if df.empty:
        print("No existen candidatos compatibles.")
        return

    print("\nCandidatos de entrada:", len(df))

    # Cada dirección se agrupa independientemente.
    grupos_finales = []

    resumen_final = []

    for direccion, grupo in df.groupby(
        "DIRECCION",
        sort=False
    ):

        grupo = grupo.copy().reset_index(drop=True)

        # Coordenadas geográficas a geometrías.
        puntos = gpd.GeoDataFrame(
            grupo,
            geometry=gpd.points_from_xy(
                grupo["LONGITUD"],
                grupo["LATITUD"]
            ),
            crs="EPSG:4326"
        )

        # Proyección métrica estimada para Bogotá.
        puntos_m = puntos.to_crs(
            puntos.estimate_utm_crs()
        )

        coordenadas = [
            [p.x, p.y]
            for p in puntos_m.geometry
        ]

        # DBSCAN agrupa puntos cercanos.
        modelo = DBSCAN(
            eps=DISTANCIA_METROS,
            min_samples=1,
            metric="euclidean"
        )

        etiquetas = modelo.fit_predict(coordenadas)

        puntos["GRUPO_ESPACIAL"] = etiquetas

        cantidad_grupos = len(set(etiquetas))

        # Construir centro de cada grupo.
        for etiqueta, subgrupo in puntos.groupby(
            "GRUPO_ESPACIAL"
        ):

            subgrupo_m = puntos_m.loc[subgrupo.index]

            centro_m = subgrupo_m.geometry.union_all().centroid

            centro_geografico = gpd.GeoSeries(
                [centro_m],
                crs=puntos_m.crs
            ).to_crs("EPSG:4326").iloc[0]

            primera_fila = subgrupo.iloc[0]

            registro = {
                "DIRECCION": direccion,
                "CODIGO_LOCALIDAD": primera_fila[
                    "CODIGO_LOCALIDAD"
                ],
                "NOMBRE_LOCALIDAD": primera_fila[
                    "NOMBRE_LOCALIDAD"
                ],
                "VIA_1": primera_fila["VIA_1"],
                "VIA_2": primera_fila["VIA_2"],
                "GRUPO_ESPACIAL": int(etiqueta),
                "PUNTOS_ORIGINALES": len(subgrupo),
                "LONGITUD_CENTRO": centro_geografico.x,
                "LATITUD_CENTRO": centro_geografico.y,
                "DISTANCIA_AGRUPACION_METROS": DISTANCIA_METROS,
                "VALIDADO": "NO"
            }

            grupos_finales.append(registro)

        resumen_final.append({
            "DIRECCION": direccion,
            "PUNTOS_ORIGINALES": len(grupo),
            "GRUPOS_ESPACIALES": cantidad_grupos,
            "DISTANCIA_METROS": DISTANCIA_METROS
        })

    # ========================================================
    # EXPORTAR
    # ========================================================

    df_grupos = pd.DataFrame(grupos_finales)

    df_resumen = pd.DataFrame(resumen_final)

    df_grupos.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    df_resumen.to_csv(
        RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULTADOS DE AGRUPACIÓN")
    print("=" * 70)

    print("Puntos originales:", len(df))

    print("Grupos espaciales:", len(df_grupos))

    print(
        "Reducción de puntos:",
        len(df) - len(df_grupos)
    )

    print("\nPrimeros grupos:")

    print(
        df_grupos.head(30).to_string(index=False)
    )

    print("\nArchivos generados:")
    print(SALIDA)
    print(RESUMEN)

    print(
        "\nATENCIÓN: los grupos son candidatos espaciales, "
        "no ubicaciones definitivas."
    )


if __name__ == "__main__":
    main()
