from pathlib import Path
import importlib.util
import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "Dataset4" / "generar_mapa_revision_georreferenciacion.py"
SPEC = importlib.util.spec_from_file_location("mapa_revision_geo_tests", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"No se pudo cargar el módulo: {SCRIPT}")
MODULO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULO)


class PruebasMapaRevision(unittest.TestCase):
    def test_normaliza_direccion_para_unir_detalle_y_resumen(self):
        self.assertEqual(
            MODULO.normalizar_direccion(" AV BOYACA-CL 170 02 "),
            "AV BOYACA-CL 170 2",
        )

    def test_mapa_conserva_etiqueta_no_validado(self):
        revision = pd.DataFrame(
            [
                {
                    "DIRECCION_NORMALIZADA": "KR 1-CL 2",
                    "CODIGO_LOCALIDAD": 7,
                    "NOMBRE_LOCALIDAD": "BOSA",
                    "GRUPO_ESPACIAL": 0,
                    "LONGITUD_CALCULADA": -74.1,
                    "LATITUD_CALCULADA": 4.6,
                    "LONGITUD_ORIGINAL": -74.1001,
                    "LATITUD_ORIGINAL": 4.6001,
                    "CANTIDAD_CRUCES": 1,
                    "ESTADO_REVISION": "CANDIDATA_UNICA_PENDIENTE_VALIDACION",
                    "VALIDADO": "NO",
                    "DISTANCIA_METROS": 1.5,
                }
            ]
        )
        detalle = pd.DataFrame(
            [
                {
                    "DIRECCION": "KR 1-CL 2",
                    "CODIGO_LOCALIDAD": 7,
                    "GRUPO_ESPACIAL": 0,
                    "LONGITUD_CRUCE": -74.1,
                    "LATITUD_CRUCE": 4.6,
                    "DISTANCIA_METROS": 1.5,
                    "OBJECTID_VIA_1": 10,
                    "OBJECTID_VIA_2": 20,
                }
            ]
        )
        localidades = gpd.GeoDataFrame(
            {
                "nombre_de_la_localidad": ["BOSA"],
                "identificador_unico_de_la_localidad": ["07"],
            },
            geometry=[Polygon([(-75, 4), (-73, 4), (-73, 5), (-75, 5)])],
            crs="EPSG:4326",
        )

        mapa = MODULO.construir_mapa(revision, detalle, localidades)
        html = mapa.get_root().render()

        self.assertIn("no validado", html.lower())
        self.assertIn("KR 1-CL 2", html)
        self.assertIn("Cruces geom", html)


if __name__ == "__main__":
    unittest.main()

