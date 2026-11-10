
import os
import math
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

RAIZ = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ / ".env")

RUTA_MODELO = RAIZ / "models" / "modelo_v8b.pkl"
RUTA_MODELO_V9D = RAIZ / "models" / "modelo_v9d_sin_diseno.pkl"

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "seguridad_vial"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}

UMBRAL_MUERTOS = float(os.getenv("MODEL_THRESHOLD", "0.30"))
CLASE_FATAL = "Con Muertos"

if not RUTA_MODELO.exists():
    raise FileNotFoundError(f"No se encontró el modelo: {RUTA_MODELO}")

modelo = joblib.load(RUTA_MODELO)
VARIABLES = list(modelo.feature_names_in_)
CLASES = [str(c) for c in modelo.classes_]
PREPROCESAMIENTO = modelo.named_steps.get("preprocesamiento")
VARIABLES_NUMERICAS: set[str] = set()
if PREPROCESAMIENTO is not None:
    for nombre_transformador, _transformador, columnas in PREPROCESAMIENTO.transformers_:
        if nombre_transformador == "numericas":
            VARIABLES_NUMERICAS = set(columnas)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    crear_tabla()
    yield


app = FastAPI(
    title="API de Seguridad Vial de Bogotá",
    description=(
        "Clasifica la gravedad de siniestros registrados con V8B y V9D. "
        "No estima la probabilidad de ocurrencia de un accidente."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# POSTGRESQL
# --------------------------------------------------

def obtener_conexion():
    return psycopg.connect(**DB_CONFIG)


def crear_tabla():
    sql = """
    CREATE TABLE IF NOT EXISTS predicciones (
        id BIGSERIAL PRIMARY KEY,
        fecha_hora TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        modelo VARCHAR(100) NOT NULL,
        prediccion VARCHAR(50) NOT NULL,
        umbral_muertos DOUBLE PRECISION,
        puntuaciones JSONB NOT NULL,
        variables_entrada JSONB NOT NULL
    );
    """
    migracion_umbral = """
    ALTER TABLE predicciones
    ALTER COLUMN umbral_muertos DROP NOT NULL;
    """

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(sql)
            cursor.execute(migracion_umbral)


# --------------------------------------------------
# MODELO Y SOLICITUD
# --------------------------------------------------

class SolicitudPrediccion(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Las 34 variables requeridas por el modelo V8B.",
    )


@app.get("/")
def inicio():
    return {
        "mensaje": "API de Seguridad Vial funcionando",
        "documentacion": "/docs",
        "estado": "/health",
    }


@app.get("/health")
def salud():
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()

        return {
            "api": "activa",
            "base_datos": "conectada",
            "modelo": "V8B",
            "modelos_disponibles": ["V8B", "V9D_sin_diseno"],
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "La API está activa, pero no pudo conectarse "
                "a PostgreSQL. Revisa el archivo .env y el servidor."
            ),
        )


@app.get("/schema")
def esquema():
    return {
        "variables_requeridas": VARIABLES,
        "cantidad_variables": len(VARIABLES),
        "clases": CLASES,
        "umbral_con_muertos": UMBRAL_MUERTOS,
    }


def guardar_prediccion(
    prediccion: str,
    puntuaciones: dict[str, float],
    variables: dict[str, Any],
    modelo_version: str = "V8B",
    umbral_muertos: float | None = UMBRAL_MUERTOS,
):
    sql = """
    INSERT INTO predicciones (
        modelo,
        prediccion,
        umbral_muertos,
        puntuaciones,
        variables_entrada
    )
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id, fecha_hora;
    """

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    modelo_version,
                    prediccion,
                    umbral_muertos,
                    Jsonb(puntuaciones),
                    Jsonb(variables),
                ),
            )
            fila = cursor.fetchone()

    return {
        "id": fila[0],
        "fecha_hora": fila[1].isoformat(),
    }


@app.post("/predict")
def predecir(solicitud: SolicitudPrediccion):
    entrada = solicitud.features

    faltantes = sorted(set(VARIABLES) - set(entrada))
    adicionales = sorted(set(entrada) - set(VARIABLES))

    if faltantes or adicionales:
        raise HTTPException(
            status_code=422,
            detail={
                "mensaje": "Las variables no coinciden con el modelo.",
                "variables_faltantes": faltantes,
                "variables_adicionales": adicionales,
            },
        )

    try:
        entrada_modelo = dict(entrada)
        for nombre in VARIABLES_NUMERICAS:
            valor = entrada[nombre]
            if valor is None:
                continue  # El imputador del pipeline trata los nulos.
            try:
                numerico = float(pd.to_numeric(valor, errors="raise"))
            except (TypeError, ValueError, OverflowError):
                raise HTTPException(
                    status_code=422,
                    detail=f"La variable {nombre} debe ser numérica o nula.",
                )
            if not math.isfinite(numerico):
                raise HTTPException(
                    status_code=422,
                    detail=f"La variable {nombre} debe ser finita o nula.",
                )
            entrada_modelo[nombre] = numerico

        datos = pd.DataFrame(
            [[entrada_modelo[nombre] for nombre in VARIABLES]],
            columns=VARIABLES,
        )

        probabilidades = modelo.predict_proba(datos)[0]
        puntuaciones = {
            str(clase): float(probabilidad)
            for clase, probabilidad in zip(CLASES, probabilidades)
        }

        if puntuaciones.get(CLASE_FATAL, 0.0) >= UMBRAL_MUERTOS:
            prediccion = CLASE_FATAL
        else:
            no_fatales = [
                clase for clase in CLASES if clase != CLASE_FATAL
            ]
            prediccion = max(
                no_fatales,
                key=lambda clase: puntuaciones[clase],
            )

        registro = guardar_prediccion(
            prediccion,
            puntuaciones,
            entrada,
        )

        return {
            "id_registro": registro["id"],
            "fecha_hora": registro["fecha_hora"],
            "modelo": "V8B",
            "prediccion": prediccion,
            "puntuaciones": puntuaciones,
            "umbral_con_muertos": UMBRAL_MUERTOS,
            "advertencia": (
                "Clasifica la gravedad condicionada a un siniestro "
                "registrado; no estima la probabilidad de que ocurra "
                "un accidente. La predicción no sustituye el criterio humano."
            ),
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error al predecir o guardar. Revisa la terminal.",
        )


@lru_cache(maxsize=1)
def cargar_modelo_v9d():
    if not RUTA_MODELO_V9D.exists():
        raise FileNotFoundError(f"No se encontró el modelo V9D: {RUTA_MODELO_V9D}")
    return joblib.load(RUTA_MODELO_V9D)


@app.get("/schema/v9d")
def esquema_v9d():
    modelo_v9d = cargar_modelo_v9d()
    variables_v9d = list(modelo_v9d.feature_names_in_)
    return {
        "modelo": "V9D_sin_diseno",
        "variables_requeridas": variables_v9d,
        "cantidad_variables": len(variables_v9d),
        "clases": [str(clase) for clase in modelo_v9d.classes_],
        "tipos_variables": {
            "ANO": "numerica",
            "MES": "numerica",
            "DIA": "numerica",
            "DIA_SEMANA": "numerica",
            "ES_FIN_DE_SEMANA": "numerica",
            "HORA_NUM": "numerica",
            "CODIGO_LOCALIDAD": "categorica (código enviado como número o texto)",
        },
        "metodo_decision": "clase con mayor probabilidad (argmax)",
        "prueba_temporal_hasta": 2020,
        "alcance": "Clasifica gravedad condicionada a un siniestro registrado.",
    }


@app.post("/predict/v9d")
def predecir_v9d(solicitud: SolicitudPrediccion):
    entrada = solicitud.features
    try:
        modelo_v9d = cargar_modelo_v9d()
        variables_v9d = list(modelo_v9d.feature_names_in_)
        clases_v9d = [str(clase) for clase in modelo_v9d.classes_]
        faltantes = sorted(set(variables_v9d) - set(entrada))
        adicionales = sorted(set(entrada) - set(variables_v9d))

        if faltantes or adicionales:
            raise HTTPException(
                status_code=422,
                detail={
                    "mensaje": "Las variables no coinciden con V9D.",
                    "variables_faltantes": faltantes,
                    "variables_adicionales": adicionales,
                },
            )

        valores = {}
        for nombre in variables_v9d:
            valor = entrada[nombre]
            if nombre == "CODIGO_LOCALIDAD":
                valores[nombre] = (
                    "DESCONOCIDO" if pd.isna(valor) else str(valor)
                )
            else:
                try:
                    valores[nombre] = pd.to_numeric(valor, errors="raise")
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=422,
                        detail=f"La variable {nombre} debe ser numérica.",
                    )

        datos = pd.DataFrame(
            [[valores[nombre] for nombre in variables_v9d]],
            columns=variables_v9d,
        )
        probabilidades = modelo_v9d.predict_proba(datos)[0]
        puntuaciones = {
            clase: float(probabilidad)
            for clase, probabilidad in zip(clases_v9d, probabilidades)
        }
        prediccion = max(puntuaciones, key=puntuaciones.get)
        registro = guardar_prediccion(
            prediccion,
            puntuaciones,
            entrada,
            modelo_version="V9D_sin_diseno",
            umbral_muertos=None,
        )
        return {
            "id_registro": registro["id"],
            "fecha_hora": registro["fecha_hora"],
            "modelo": "V9D_sin_diseno",
            "prediccion": prediccion,
            "puntuaciones": puntuaciones,
            "metodo_decision": "clase con mayor probabilidad (argmax)",
            "advertencia": (
                "Clasifica la gravedad entre siniestros registrados; no estima "
                "la probabilidad de que ocurra un accidente. La clase fatal "
                "tiene precisión baja en la evaluación temporal. Años posteriores "
                "a 2020 están fuera del periodo evaluado."
            ),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error al predecir o guardar con V9D. Revisa la terminal.",
        )


@app.get("/predicciones/recientes")
def predicciones_recientes(limite: int = 10):
    if limite < 1 or limite > 100:
        raise HTTPException(
            status_code=422,
            detail="El límite debe estar entre 1 y 100.",
        )

    sql = """
    SELECT id, fecha_hora, modelo, prediccion,
           umbral_muertos, puntuaciones
    FROM predicciones
    ORDER BY fecha_hora DESC
    LIMIT %s;
    """

    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(sql, (limite,))
                filas = cursor.fetchall()

        return {
            "cantidad": len(filas),
            "predicciones": [
                {
                    "id": f[0],
                    "fecha_hora": f[1].isoformat(),
                    "modelo": f[2],
                    "prediccion": f[3],
                    "umbral_muertos": f[4],
                    "puntuaciones": f[5],
                }
                for f in filas
            ],
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error consultando las predicciones en PostgreSQL.",
        )
