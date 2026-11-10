
from pathlib import Path
import shutil
import pandas as pd

# ==========================================================
# CONFIGURACION GENERAL
# ==========================================================

BASE = Path(__file__).resolve().parents[2]

DESTINO = BASE / "ENTREGA_POWER_BI"

CARPETAS = [
    "01_DATOS_HISTORICOS",
    "02_ANALISIS_GRAVEDAD",
    "03_GEOGRAFIA_CANDIDATA",
    "04_MODELO_V8B",
    "05_DOCUMENTACION",
    "06_EXPERIMENTO_PREVENTIVO",
]

for carpeta in CARPETAS:
    (DESTINO / carpeta).mkdir(
        parents=True,
        exist_ok=True
    )

# Inventario de archivos
inventario = []


# ==========================================================
# FUNCION PARA COPIAR ARCHIVOS
# ==========================================================

def copiar_archivo(ruta_origen, carpeta_destino, obligatorio=True):

    origen = BASE / ruta_origen
    destino = DESTINO / carpeta_destino / origen.name

    registro = {
        "ARCHIVO": origen.name,
        "ORIGEN": str(origen),
        "DESTINO": str(destino),
        "ESTADO": "",
        "TAMANO_MB": None,
        "OBLIGATORIO": "SI" if obligatorio else "NO"
    }

    if not origen.exists():
        registro["ESTADO"] = "NO ENCONTRADO"
        inventario.append(registro)

        print(f"[FALTA] {origen.name}")
        return False

    destino.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(origen, destino)

    tamano = destino.stat().st_size / (1024 ** 2)

    registro["ESTADO"] = "COPIADO"
    registro["TAMANO_MB"] = round(tamano, 2)

    inventario.append(registro)

    print(
        f"[OK] {origen.name} "
        f"({tamano:.2f} MB)"
    )

    return True


# ==========================================================
# 1. DATOS HISTORICOS
# ==========================================================

print("\n==============================================")
print(" 1. DATOS HISTORICOS")
print("==============================================")

archivos_historicos = [
    "models/powerbi_gravedad_localidad.csv",
    "models/powerbi_gravedad_hora.csv",
    "models/powerbi_gravedad_dia.csv",
    "models/powerbi_localidad_hora.csv",
    "models/powerbi_evolucion_anual.csv",
    "data/processed/siniestros_integrados.csv",
    "data/processed/siniestros_modelo.csv",
]

for archivo in archivos_historicos:
    copiar_archivo(
        archivo,
        "01_DATOS_HISTORICOS"
    )


# ==========================================================
# 2. ANALISIS DE GRAVEDAD
# ==========================================================

print("\n==============================================")
print(" 2. ANALISIS DE GRAVEDAD")
print("==============================================")

archivos_gravedad = [
    "models/prioridad_gravedad_por_localidad.csv",
    "models/prioridad_gravedad_por_hora.csv",
    "models/prioridad_gravedad_por_dia.csv",
    "models/evolucion_anual_gravedad.csv",
]

for archivo in archivos_gravedad:
    copiar_archivo(
        archivo,
        "02_ANALISIS_GRAVEDAD"
    )


# ==========================================================
# 3. GEOGRAFIA CANDIDATA
# ==========================================================

print("\n==============================================")
print(" 3. GEOGRAFIA CANDIDATA")
print("==============================================")

archivos_geograficos = [
    "models/candidatos_agrupados_espacialmente.csv",
    "models/auditoria_georreferenciacion_50_grupos.csv",
    "models/detalle_cruces_geometricos_50_grupos.csv",
    "models/revision_calidad_georreferenciacion.csv",
    "models/auditoria_cruces_normalizados.csv",
    "models/resumen_cruces_normalizados.csv",
    "models/mapa_intersecciones_candidatas.html",
    "models/hoja_revision_manual_georreferenciacion.csv",
    "models/mapa_revision_manual_georreferenciacion.html",
    "data/raw/Malla_Vial_Integral_Bogota_D_C.geojson",
    "data/raw/poligonos-localidad-2.geojson",
]

for archivo in archivos_geograficos:
    copiar_archivo(
        archivo,
        "03_GEOGRAFIA_CANDIDATA",
        obligatorio=False
    )


# ==========================================================
# 4. MODELO V8B
# ==========================================================

print("\n==============================================")
print(" 4. MODELO V8B")
print("==============================================")

# Buscar resultados CSV y JSON relacionados con V8B.
# No copiamos el modelo .pkl porque Power BI no lo necesita.

carpeta_models = BASE / "models"

patrones_modelo = [
    "*v8b*.csv",
    "*v8b*.json",
]

archivos_modelo = set()

for patron in patrones_modelo:
    archivos_modelo.update(
        carpeta_models.glob(patron)
    )

if archivos_modelo:
    for archivo in sorted(archivos_modelo):

        copiar_archivo(
            archivo.relative_to(BASE),
            "04_MODELO_V8B",
            obligatorio=False
        )
else:
    print(
        "[AVISO] No se encontraron archivos adicionales "
        "de métricas o importancia."
    )


# ==========================================================
# 5. DIAGNOSTICO DEL EXPERIMENTO PREVENTIVO
# ==========================================================

print("\n==============================================")
print(" 5. DIAGNOSTICO EXPERIMENTO PREVENTIVO")
print("==============================================")

archivos_diagnostico = [
    "models/diagnostico_panel_interseccion_hora.json",
    "models/diagnostico_panel_anual.csv",
    "models/diagnostico_panel_split_temporal.csv",
    "models/diagnostico_panel_geografia.csv",
    "models/diagnostico_panel_horas.csv",
]

for archivo in archivos_diagnostico:
    copiar_archivo(
        archivo,
        "06_EXPERIMENTO_PREVENTIVO",
        obligatorio=False,
    )


# ==========================================================
# 5. DOCUMENTACION
# ==========================================================

print("\n==============================================")
print(" 5. DOCUMENTACION")
print("==============================================")

documentacion = """
ENTREGA POWER BI
PROYECTO: SISTEMA INTELIGENTE PARA LA PREDICCION
DEL NIVEL DE RIESGO DE ACCIDENTES DE TRANSITO EN BOGOTA

CONTENIDO

01_DATOS_HISTORICOS
Bases históricas de siniestros y tablas agregadas.

02_ANALISIS_GRAVEDAD
Distribución de gravedad por localidad, hora, día y año.

03_GEOGRAFIA_CANDIDATA
Intersecciones candidatas, cruces geométricos,
malla vial y polígonos de localidades.

04_MODELO_V8B
Métricas, resultados e importancia de variables
del clasificador V8B, cuando estén disponibles.

05_DOCUMENTACION
Inventario y notas de uso.

06_EXPERIMENTO_PREVENTIVO
Diagnóstico del panel intersección × fecha × hora y tablas
de cobertura temporal/geográfica para revisión descriptiva.

ALCANCE Y PRECAUCIONES

1. La base histórica contiene 196.152 siniestros
   registrados entre 2015 y 2020.

2. Las tablas históricas permiten análisis descriptivo.

3. Las 50 intersecciones disponibles son candidatas.
   No deben presentarse como ubicaciones verificadas
   de todos los accidentes.

4. Todavía no se dispone de coordenadas verificadas
   para los 196.152 registros.

5. Los registros históricos describen cantidades y
   proporciones, no tasas de riesgo por exposición
   al tránsito.

6. El modelo V8B clasifica la gravedad de accidentes
   ya registrados. No predice la ocurrencia futura
   de accidentes.

7. Las métricas del modelo deben mostrarse con sus
   limitaciones y su contexto de evaluación.

8. No compartir contraseñas, archivos .env ni
   credenciales de la base de datos.

9. El diagnóstico del panel preventivo no encontró una base
   válida para entrenar o evaluar todavía: las 50 ubicaciones
   tienen coordenadas calculadas, pero 0 cuentan con validación
   humana; 43 son ambiguas y 7 siguen pendientes.

10. Las coincidencias por dirección y localidad cubren 2.035
    registros (1,04 % del histórico) y son preliminares. Los
    huecos del panel significan “sin reporte registrado”; no son
    negativos confirmados por falta de universo espacial validado,
    cobertura demostrada y datos de exposición al tránsito.

11. El split 2015–2018 / 2019 / 2020 está definido, pero las
    métricas PR-AUC, Brier y recall@K no se calcularon. V8B y V9D
    no son predictores preventivos de ocurrencia.

PAGINAS PROPUESTAS

1. Panorama general.
2. Análisis geográfico.
3. Análisis temporal.
4. Gravedad y características.
5. Resultados del modelo V8B con sus limitaciones.
6. Estado del experimento preventivo (diagnóstico descriptivo).

IMPORTANTE

Los archivos que no se encontraron se relacionan
en el inventario_entrega_powerbi.csv.
"""

archivo_readme = (
    DESTINO / "05_DOCUMENTACION" / "LEEME.txt"
)

archivo_readme.write_text(
    documentacion.strip(),
    encoding="utf-8"
)

print("[OK] LEEME.txt generado")


# ==========================================================
# GENERAR INVENTARIO
# ==========================================================

print("\n==============================================")
print(" GENERANDO INVENTARIO")
print("==============================================")

df_inventario = pd.DataFrame(inventario)

archivo_inventario = (
    DESTINO / "05_DOCUMENTACION"
    / "inventario_entrega_powerbi.csv"
)

df_inventario.to_csv(
    archivo_inventario,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# RESUMEN FINAL
# ==========================================================

copiados = (
    df_inventario["ESTADO"] == "COPIADO"
).sum()

faltantes = (
    df_inventario["ESTADO"] == "NO ENCONTRADO"
).sum()

tamano_total = df_inventario.loc[
    df_inventario["ESTADO"] == "COPIADO",
    "TAMANO_MB"
].sum()

print("\n==============================================")
print(" ENTREGA POWER BI PREPARADA")
print("==============================================")

print(f"\nArchivos copiados: {copiados}")
print(f"Archivos no encontrados: {faltantes}")
print(f"Tamaño total: {tamano_total:.2f} MB")

print("\nUbicacion de la entrega:")
print(DESTINO)

print("\nInventario:")
print(archivo_inventario)

print("\nEstado de los archivos:")
print(
    df_inventario[
        ["ARCHIVO", "ESTADO", "OBLIGATORIO"]
    ].to_string(index=False)
)

print("\nProceso terminado.")

