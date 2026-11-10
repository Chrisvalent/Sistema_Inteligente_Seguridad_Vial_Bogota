import pandas as pd
from pathlib import Path

# ============================================================
# RUTAS
# ============================================================

RUTA_BASE = Path(__file__).resolve().parents[2]

RUTA_EXCEL = (
    RUTA_BASE
    / "data"
    / "raw"
    / "siniestros_viales_consolidados_bogota_dc.xlsx"
)

# ============================================================
# CARGAR DICCIONARIO
# ============================================================

print("=" * 70)
print("AUDITORÍA DEL DICCIONARIO ORIGINAL")
print("=" * 70)

diccionario = pd.read_excel(
    RUTA_EXCEL,
    sheet_name="DICCIONARIO"
)

diccionario.columns = (
    diccionario.columns
    .str.strip()
    .str.upper()
)

# ============================================================
# CONSULTAR CLASE Y GRAVEDAD
# ============================================================

for campo in ["CLASE", "GRAVEDAD"]:

    print(f"\n{'=' * 60}")
    print(f"DICCIONARIO: {campo}")
    print("=" * 60)

    resultado = diccionario[
        (diccionario["HOJA"] == "SINIESTROS")
        & (diccionario["CAMPO"] == campo)
    ][["CODIGO", "DESCRIPCION"]].drop_duplicates()

    print(resultado.to_string(index=False))

    print(f"\nCantidad de códigos: {len(resultado)}")

# ============================================================
# VERIFICAR ORIGEN EN SINIESTROS
# ============================================================

siniestros = pd.read_excel(
    RUTA_EXCEL,
    sheet_name="SINIESTROS"
)

siniestros.columns = (
    siniestros.columns
    .str.strip()
    .str.upper()
)

print("\n" + "=" * 70)
print("VALORES ORIGINALES DE CLASE")
print("=" * 70)

print(
    siniestros["CLASE"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAuditoría finalizada.")