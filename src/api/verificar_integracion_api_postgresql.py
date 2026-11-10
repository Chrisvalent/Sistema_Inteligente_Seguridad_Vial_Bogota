"""Verificación real y de solo lectura de API + PostgreSQL.

Comprueba /health, los esquemas de ambos modelos y la persistencia conocida
de los registros técnicos 8 y 9. No envía POST ni modifica la base de datos.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.main import obtener_conexion  # noqa: E402


API_BASE = "http://127.0.0.1:8000"


def get_json(route: str) -> dict:
    request = urllib.request.Request(f"{API_BASE}{route}", method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{route}: HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def text_tree(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{text_tree(k)} {text_tree(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(text_tree(item) for item in value)
    return str(value)


def main() -> None:
    health = get_json("/health")
    schema_v8b = get_json("/schema")
    schema_v9d = get_json("/schema/v9d")
    assert health.get("api") == "activa"
    assert health.get("base_datos") == "conectada"
    assert schema_v8b.get("cantidad_variables") == 34
    assert schema_v9d.get("cantidad_variables") == 7

    with obtener_conexion() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM predicciones")
            total_rows = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT id, modelo, prediccion, fecha_hora, variables_entrada "
                "FROM predicciones WHERE id IN (8, 9) ORDER BY id"
            )
            rows = cursor.fetchall()

    if [int(row[0]) for row in rows] != [8, 9]:
        raise AssertionError("No se encontraron los dos registros técnicos esperados (IDs 8 y 9).")

    technical = []
    for row_id, model, label, created_at, features in rows:
        content = text_tree(features).upper()
        year = features.get("ANO") if isinstance(features, dict) else None
        if int(row_id) == 8:
            classification = "TECNICO_MARCADO"
            if "PRUEBA" not in content:
                raise AssertionError("El registro 8 perdió el marcador técnico esperado.")
        else:
            classification = "SINTETICO_FUERA_DEL_PERIODO" if year == 2026 else "REVISAR"
            if classification != "SINTETICO_FUERA_DEL_PERIODO":
                raise AssertionError("El registro 9 no conserva el año sintético esperado (2026).")
        technical.append({
            "id": int(row_id),
            "modelo": model,
            "prediccion_guardada": label,
            "fecha_creacion": created_at.isoformat(),
            "clasificacion_auditoria": classification,
            "ano_entrada": year,
        })

    print(json.dumps({
        "resultado": "OK_SOLO_LECTURA",
        "api": {
            "health_http_200": True,
            "postgres_conectado": True,
            "variables_schema_v8b": schema_v8b["cantidad_variables"],
            "variables_schema_v9d": schema_v9d["cantidad_variables"],
        },
        "postgres": {
            "filas_actuales": total_rows,
            "registros_tecnicos_a_excluir_de_resultados": technical,
            "filas_insertadas_o_modificadas_por_esta_prueba": 0,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
