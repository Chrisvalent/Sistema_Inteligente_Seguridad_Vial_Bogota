from pathlib import Path
import importlib.util
import unittest

import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "Dataset4" / "preparar_hoja_revision_georreferenciacion.py"
SPEC = importlib.util.spec_from_file_location("hoja_revision_geo_tests", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"No se pudo cargar el módulo: {SCRIPT}")
MODULO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULO)


class PruebasHojaRevisionGeorreferenciacion(unittest.TestCase):
    def test_comprueba_pertenencia_al_poligono_de_localidad(self):
        poligonos = {"07": Polygon([(-75, 4), (-73, 4), (-73, 5), (-75, 5)])}
        fila = pd.Series(
            {
                "LONGITUD_CALCULADA": -74.0,
                "LATITUD_CALCULADA": 4.5,
                "CODIGO_LOCALIDAD": 7,
            }
        )

        self.assertEqual(
            MODULO.coordenada_en_localidad(fila, poligonos), "SI"
        )
        fila["LONGITUD_CALCULADA"] = -72.0
        self.assertEqual(
            MODULO.coordenada_en_localidad(fila, poligonos), "NO"
        )

    def test_multiples_grupos_siguen_ambiguos_aunque_haya_un_cruce(self):
        estado = MODULO.clasificar_revision(
            pd.Series(
                {
                    "LONGITUD_CALCULADA": -74.1,
                    "LATITUD_CALCULADA": 4.6,
                    "CANTIDAD_CRUCES": 1,
                    "TOTAL_GRUPOS_DIRECCION": 2,
                }
            )
        )

        self.assertEqual(estado, "AMBIGUA_REQUIERE_REVISION")

    def test_cruce_unico_y_una_sola_direccion_sigue_pendiente(self):
        estado = MODULO.clasificar_revision(
            pd.Series(
                {
                    "LONGITUD_CALCULADA": -74.1,
                    "LATITUD_CALCULADA": 4.6,
                    "CANTIDAD_CRUCES": 1,
                    "TOTAL_GRUPOS_DIRECCION": 1,
                }
            )
        )

        self.assertEqual(estado, "CANDIDATA_UNICA_PENDIENTE_VALIDACION")

    def test_datos_incompletos_no_se_tratan_como_cruce_unico(self):
        estado = MODULO.clasificar_revision(
            pd.Series(
                {
                    "LONGITUD_CALCULADA": -74.1,
                    "LATITUD_CALCULADA": 4.6,
                    "CANTIDAD_CRUCES": float("nan"),
                    "TOTAL_GRUPOS_DIRECCION": 1,
                }
            )
        )

        self.assertEqual(estado, "DATOS_INCOMPLETOS_REQUIERE_REVISION")

    def test_no_aprueba_automaticamente_ninguna_fila(self):
        calidad = pd.DataFrame(
            [
                {
                    "DIRECCION_NORMALIZADA": "KR 1-CL 2",
                    "CODIGO_LOCALIDAD": 1,
                    "GRUPO_ESPACIAL": 0,
                    "LONGITUD_CALCULADA": -74.1,
                    "LATITUD_CALCULADA": 4.6,
                    "CANTIDAD_CRUCES": 1,
                    "TOTAL_GRUPOS_DIRECCION": 1,
                    "VALIDADO": "NO",
                }
            ]
        )
        resumen = pd.DataFrame(
            [
                {
                    "DIRECCION_NORMALIZADA": "KR 1-CL 2",
                    "CODIGO_LOCALIDAD": 1,
                    "CANTIDAD_REGISTROS_CRUCE": 5,
                }
            ]
        )

        resultado = MODULO.construir_hoja(calidad, resumen)

        self.assertEqual(resultado.loc[0, "VALIDADO"], "NO")
        self.assertEqual(
            resultado.loc[0, "ESTADO_REVISION"],
            "CANDIDATA_UNICA_PENDIENTE_VALIDACION",
        )
        self.assertEqual(resultado.loc[0, "LONGITUD_APROBADA"], "")


if __name__ == "__main__":
    unittest.main()
