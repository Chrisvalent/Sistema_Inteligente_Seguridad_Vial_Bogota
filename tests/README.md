# Pruebas automatizadas

Desde la raíz del proyecto, ejecuta:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Las pruebas cubren configuración, métricas y funciones del modelo V9D; la comparación de baselines sobre la misma partición temporal; la normalización de direcciones y generación del mapa de revisión; la clasificación de ambigüedad geográfica; y validación/predicción de la API.

La prueba de API sustituye la función que guarda en PostgreSQL y evita cargar `.env`; no inserta datos ni necesita una base de datos activa. La prueba de predicción requiere `models/modelo_v8b.pkl` y `data/processed/siniestros_modelo.csv`.

## Integración real con PostgreSQL

La conexión real se verificó el 7 de noviembre de 2026 contra `seguridad_vial`. Quedaron dos registros de prueba al final de `public.predicciones`: ID 8, marcado con `__PRUEBA_TECNICA_API_20261107__`, y ID 9, escenario sintético V9D con fecha de ejemplo 2026. No representan siniestros reales ni deben usarse para análisis.

`src/api/probar_guardado_postgresql.py` ejecuta otra inserción V8B marcada y persistente cada vez que se corre; usarla solo cuando se quiera crear un registro de prueba. No borra filas.
