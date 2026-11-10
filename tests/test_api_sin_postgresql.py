from pathlib import Path
import asyncio
import importlib.util
import os
import unittest
from unittest.mock import patch

import dotenv
import httpx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_SCRIPT = ROOT / "src" / "api" / "main.py"
DATASET = ROOT / "data" / "processed" / "siniestros_modelo.csv"


class PruebasAPISinPostgreSQL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Evitar leer .env o usar credenciales del entorno durante las pruebas.
        cls.dotenv_patch = patch.object(
            dotenv, "load_dotenv", lambda *args, **kwargs: False
        )
        cls.dotenv_patch.start()
        cls.env_patch = patch.dict(
            os.environ,
            {
                "PGHOST": "localhost",
                "PGPORT": "5432",
                "PGDATABASE": "seguridad_vial_test",
                "PGUSER": "postgres",
                "PGPASSWORD": "",
                "MODEL_THRESHOLD": "0.30",
            },
        )
        cls.env_patch.start()

        spec = importlib.util.spec_from_file_location(
            "api_main_isolated_tests", API_SCRIPT
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"No se pudo cargar la API: {API_SCRIPT}")
        cls.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.api)

    @classmethod
    def tearDownClass(cls):
        cls.env_patch.stop()
        cls.dotenv_patch.stop()

    def solicitar_sin_postgresql(self, payload):
        with patch.object(self.api, "crear_tabla") as crear_tabla:
            with patch.object(
                self.api,
                "guardar_prediccion",
                return_value={"id": 123, "fecha_hora": "prueba"},
            ) as guardar:
                async def enviar():
                    async with self.api.lifespan(self.api.app):
                        transporte = httpx.ASGITransport(app=self.api.app)
                        async with httpx.AsyncClient(
                            transport=transporte,
                            base_url="http://test",
                        ) as cliente:
                            return await cliente.post("/predict", json=payload)

                respuesta = asyncio.run(enviar())

        return respuesta, crear_tabla.call_count, guardar.call_count

    def test_rechaza_variables_faltantes(self):
        respuesta, inicializaciones, _ = self.solicitar_sin_postgresql(
            {"features": {}}
        )

        self.assertEqual(respuesta.status_code, 422)
        self.assertEqual(inicializaciones, 1)

    def test_rechaza_variables_adicionales(self):
        features = {variable: 1 for variable in self.api.VARIABLES}
        features["VARIABLE_INVENTADA"] = 1

        respuesta, inicializaciones, escrituras = self.solicitar_sin_postgresql(
            {"features": features}
        )

        self.assertEqual(respuesta.status_code, 422)
        self.assertEqual(escrituras, 0)
        self.assertEqual(inicializaciones, 1)

    def test_predice_con_fila_historica_sin_escribir_en_db(self):
        datos = pd.read_csv(DATASET, usecols=self.api.VARIABLES, nrows=100)
        filas = datos.dropna(subset=self.api.VARIABLES)
        self.assertFalse(filas.empty, "No hay filas completas para probar la API.")
        features = filas.iloc[0].to_dict()

        respuesta, inicializaciones, escrituras = self.solicitar_sin_postgresql(
            {"features": features}
        )

        self.assertEqual(respuesta.status_code, 200)
        resultado = respuesta.json()
        self.assertEqual(resultado["id_registro"], 123)
        self.assertEqual(resultado["modelo"], "V8B")
        self.assertAlmostEqual(sum(resultado["puntuaciones"].values()), 1.0)
        self.assertIn("no estima la probabilidad", resultado["advertencia"])
        self.assertEqual(escrituras, 1)
        self.assertEqual(inicializaciones, 1)

    def test_v8b_rechaza_valor_no_numerico_con_422(self):
        datos = pd.read_csv(DATASET, usecols=self.api.VARIABLES, nrows=100)
        fila = datos.dropna(subset=self.api.VARIABLES).iloc[0].to_dict()
        fila["ANO"] = "dato_invalido"

        respuesta, _, escrituras = self.solicitar_sin_postgresql(
            {"features": fila}
        )

        self.assertEqual(respuesta.status_code, 422)
        self.assertIn("ANO debe ser numérica", respuesta.json()["detail"])
        self.assertEqual(escrituras, 0)

    def test_inicio_de_api_inicializa_tabla_en_lifespan(self):
        async def ejecutar_lifespan():
            with patch.object(self.api, "crear_tabla") as crear_tabla:
                async with self.api.lifespan(self.api.app):
                    pass
                crear_tabla.assert_called_once_with()

        asyncio.run(ejecutar_lifespan())

    def test_v9d_rechaza_variables_ajenas_al_modelo(self):
        class ModeloV9DPrueba:
            feature_names_in_ = np.array(["ANO", "CODIGO_LOCALIDAD"])
            classes_ = np.array(["Con Heridos", "Con Muertos", "Solo Daños"])

        async def enviar():
            with patch.object(self.api, "crear_tabla"):
                with patch.object(
                    self.api, "cargar_modelo_v9d", return_value=ModeloV9DPrueba()
                ):
                    async with self.api.lifespan(self.api.app):
                        transporte = httpx.ASGITransport(app=self.api.app)
                        async with httpx.AsyncClient(
                            transport=transporte, base_url="http://test"
                        ) as cliente:
                            return await cliente.post(
                                "/predict/v9d",
                                json={"features": {"ANO": 2020, "EXTRA": 1}},
                            )

        respuesta = asyncio.run(enviar())
        self.assertEqual(respuesta.status_code, 422)

    def test_v9d_usa_argmax_y_guarda_version_sin_umbral(self):
        class ModeloV9DPrueba:
            feature_names_in_ = np.array(["ANO", "CODIGO_LOCALIDAD"])
            classes_ = np.array(["Con Heridos", "Con Muertos", "Solo Daños"])

            def predict_proba(self, datos):
                if datos["CODIGO_LOCALIDAD"].iloc[0] != "7":
                    raise AssertionError("La localidad debe normalizarse a texto.")
                return np.array([[0.7, 0.2, 0.1]])

        features = {"ANO": 2020, "CODIGO_LOCALIDAD": 7}

        async def enviar():
            with patch.object(self.api, "crear_tabla"):
                with patch.object(
                    self.api, "cargar_modelo_v9d", return_value=ModeloV9DPrueba()
                ):
                    with patch.object(
                        self.api,
                        "guardar_prediccion",
                        return_value={"id": 456, "fecha_hora": "prueba"},
                    ) as guardar:
                        async with self.api.lifespan(self.api.app):
                            transporte = httpx.ASGITransport(app=self.api.app)
                            async with httpx.AsyncClient(
                                transport=transporte, base_url="http://test"
                            ) as cliente:
                                respuesta = await cliente.post(
                                    "/predict/v9d", json={"features": features}
                                )
                        guardar.assert_called_once_with(
                            "Con Heridos",
                            {
                                "Con Heridos": 0.7,
                                "Con Muertos": 0.2,
                                "Solo Daños": 0.1,
                            },
                            features,
                            modelo_version="V9D_sin_diseno",
                            umbral_muertos=None,
                        )
            return respuesta

        respuesta = asyncio.run(enviar())
        self.assertEqual(respuesta.status_code, 200)
        resultado = respuesta.json()
        self.assertEqual(resultado["modelo"], "V9D_sin_diseno")
        self.assertEqual(resultado["metodo_decision"], "clase con mayor probabilidad (argmax)")


if __name__ == "__main__":
    unittest.main()
