
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


BASE = Path(__file__).resolve().parents[2]

ENTRADA = (
    BASE / "models" / "candidatos_filtrados_localidad.csv"
)

SALIDA = (
    BASE / "models" / "auditoria_calidad_grupos.csv"
)


def main():

    print("=" * 70)
    print("AUDITORÍA DE CALIDAD DE LOS GRUPOS ESPACIALES")
    print("=" * 70)

    df = pd.read_csv(ENTRADA, encoding="utf-8-sig")

    df = df[
        df["DENTRO_LOCALIDAD"].astype(str).str.lower()
        .isin(["true", "1"])
    ].copy()

    df["geometry"] = [
        Point(x, y)
        for x, y in zip(df["LONGITUD"], df["LATITUD"])
    ]

    geo = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:4326"
    )

    resultados = []

    for direccion, grupo in geo.groupby("DIRECCION"):

        grupo = grupo.reset_index(drop=True)

        grupo_m = grupo.to_crs(grupo.estimate_utm_crs())

        coords = [
            (p.x, p.y)
            for p in grupo_m.geometry
        ]

        # Agrupación reproducible para la auditoría.
        from sklearn.cluster import DBSCAN

        etiquetas = DBSCAN(
            eps=20,
            min_samples=1
        ).fit_predict(coords)

        grupo["GRUPO"] = etiquetas

        for etiqueta, subgrupo in grupo.groupby("GRUPO"):

            indices = subgrupo.index.tolist()

            puntos = [
                grupo_m.geometry.iloc[i]
                for i in indices
            ]

            centro = subgrupo.to_crs(
                grupo_m.crs
            ).geometry.union_all().centroid

            distancias = [
                punto.distance(centro)
                for punto in puntos
            ]

            max_distancia = max(distancias) if distancias else 0

            # Distancia máxima entre cualquier pareja.
            diametro = 0

            for i in range(len(puntos)):
                for j in range(i + 1, len(puntos)):
                    diametro = max(
                        diametro,
                        puntos[i].distance(puntos[j])
                    )

            if diametro <= 20:
                calidad = "COMPACTO"
            elif diametro <= 40:
                calidad = "REVISAR DISPERSION"
            else:
                calidad = "DISPERSION ALTA"

            resultados.append({

                "DIRECCION": direccion,

                "CODIGO_LOCALIDAD": subgrupo.iloc[0][
                    "CODIGO_LOCALIDAD"
                ],

                "GRUPO": int(etiqueta),

                "PUNTOS": len(subgrupo),

                "DIAMETRO_METROS": round(diametro, 2),

                "DISTANCIA_MAX_CENTRO_METROS": round(
                    max_distancia, 2
                ),

                "CALIDAD_GEOMETRICA": calidad

            })

    salida = pd.DataFrame(resultados)

    salida = salida.sort_values(
        ["CALIDAD_GEOMETRICA", "DIAMETRO_METROS"],
        ascending=[True, False]
    )

    salida.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nGrupos auditados:", len(salida))

    print("\nClasificación geométrica:")

    print(
        salida["CALIDAD_GEOMETRICA"]
        .value_counts()
        .to_string()
    )

    print("\nGrupos con mayor dispersión:")

    print(
        salida.sort_values(
            "DIAMETRO_METROS",
            ascending=False
        ).head(15).to_string(index=False)
    )

    print("\nArchivo generado:")
    print(SALIDA)

    print("\nAuditoría terminada.")


if __name__ == "__main__":
    main()
