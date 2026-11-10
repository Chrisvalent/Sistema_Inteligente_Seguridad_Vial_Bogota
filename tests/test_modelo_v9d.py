from pathlib import Path
import importlib.util
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "Dataset3" / "entrenar_modelo_v9d.py"


def cargar_modulo_v9d():
    spec = importlib.util.spec_from_file_location("modelo_v9d_tests", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el módulo: {SCRIPT}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class PruebasModeloV9D(unittest.TestCase):
    def test_excluye_diseno_lugar_hasta_validar_su_disponibilidad(self):
        modulo = cargar_modulo_v9d()

        self.assertFalse(modulo.INCLUIR_DISENO_LUGAR)

    def test_normalizar_quita_acentos_y_espacios(self):
        modulo = cargar_modulo_v9d()

        self.assertEqual(modulo.normalizar("  Bogotá "), "BOGOTA")


    def test_metricas_de_clase_con_muertos(self):
        modulo = cargar_modulo_v9d()
        precision, recall, f1 = modulo.evaluar_muertos(
            pd.Series(["Con Muertos", "Solo Daños", "Con Muertos"]),
            pd.Series(["Con Muertos", "Con Muertos", "Solo Daños"]),
            "Con Muertos",
        )

        self.assertEqual((precision, recall, f1), (0.5, 0.5, 0.5))


    def test_pipeline_entrena_con_variables_temporales_y_localidad(self):
        modulo = cargar_modulo_v9d()
        modelo = modulo.crear_modelo(
            numericas=["ANO", "MES"],
            categoricas=["LOCALIDAD"],
            peso_clases={"Con Heridos": 1.0, "Solo Daños": 1.0},
            arboles=3,
        )
        X = pd.DataFrame(
            {
                "ANO": [2015, 2016, 2017, 2018],
                "MES": [1, 2, 3, 4],
                "LOCALIDAD": ["1", "1", "2", "2"],
            }
        )
        y = pd.Series(["Solo Daños", "Con Heridos", "Solo Daños", "Con Heridos"])

        modelo.fit(X, y)

        self.assertEqual(len(modelo.predict(X)), len(X))


if __name__ == "__main__":
    unittest.main()
