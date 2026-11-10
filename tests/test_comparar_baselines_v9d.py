from pathlib import Path
import importlib.util
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "Dataset4" / "comparar_baselines_v9d.py"
SPEC = importlib.util.spec_from_file_location("comparar_baselines_tests", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"No se pudo cargar el módulo: {SCRIPT}")
MODULO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULO)


class PruebasComparacionBaselines(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "CODIGO_LOCALIDAD": ["01", "01", "01", "01", "02", "02", "02", "02"],
                "GRAVEDAD": [
                    "Solo Daños", "Solo Daños", "Solo Daños", "Con Muertos",
                    "Con Heridos", "Con Heridos", "Con Heridos", "Solo Daños",
                ],
            }
        )
        self.test = pd.DataFrame(
            {
                "ANO": [2019, 2019, 2020],
                "CODIGO_LOCALIDAD": ["01", "02", "03"],
                "GRAVEDAD": ["Con Muertos", "Con Heridos", "Solo Daños"],
            }
        )
        self.predicciones = pd.DataFrame(
            {
                "GRAVEDAD_REAL": ["Con Muertos", "Con Heridos", "Solo Daños"],
                "GRAVEDAD_PREDICHA": ["Con Muertos", "Solo Daños", "Solo Daños"],
            }
        )

    def test_calcula_baselines_en_la_misma_particion(self):
        metricas, detalle = MODULO.comparar(
            self.train, self.test, self.predicciones
        )

        self.assertEqual(len(detalle), 3)
        self.assertEqual(metricas["mayoria_global"]["clase_predicha"], "Solo Daños")
        self.assertEqual(
            detalle["PREDICCION_MODA_LOCALIDAD"].tolist(),
            ["Solo Daños", "Con Heridos", "Solo Daños"],
        )
        self.assertGreater(
            metricas["V9D_sin_diseno"]["recall_con_muertos"], 0
        )

    def test_rechaza_predicciones_desalineadas(self):
        desalineadas = self.predicciones.iloc[::-1].reset_index(drop=True)

        with self.assertRaises(ValueError):
            MODULO.comparar(self.train, self.test, desalineadas)


if __name__ == "__main__":
    unittest.main()
