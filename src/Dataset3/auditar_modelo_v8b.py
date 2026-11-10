
from pathlib import Path
import hashlib
import sys
import platform
import joblib
import sklearn
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]
RUTA_MODELO = RAIZ / "models" / "modelo_v8b.pkl"
RUTA_REPORTE = RAIZ / "models" / "auditoria_modelo_v8b.txt"


def calcular_sha256(ruta):
    sha256 = hashlib.sha256()

    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(8192), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


def inspeccionar_objeto(nombre, objeto):
    print(f"\n--- {nombre} ---")
    print("Tipo:", type(objeto).__name__)
    print("Clase completa:", type(objeto))

    if hasattr(objeto, "n_features_in_"):
        print("Número de variables:", objeto.n_features_in_)

    if hasattr(objeto, "feature_names_in_"):
        variables = list(objeto.feature_names_in_)
        print("\nVariables del modelo:")
        for i, variable in enumerate(variables, start=1):
            print(f"{i:02d}. {variable}")

    if hasattr(objeto, "classes_"):
        print("\nClases:")
        print(list(objeto.classes_))

    if hasattr(objeto, "get_params"):
        parametros = objeto.get_params(deep=False)
        print("\nParámetros principales:")

        for clave, valor in parametros.items():
            print(f"{clave}: {valor}")


def main():
    print("=" * 65)
    print("AUDITORÍA TÉCNICA DEL MODELO V8B")
    print("=" * 65)

    if not RUTA_MODELO.exists():
        print("\nERROR: No se encontró el archivo:")
        print(RUTA_MODELO)
        return

    print("\n1. IDENTIFICACIÓN DEL ARCHIVO")
    print("Ruta:", RUTA_MODELO)
    print("Tamaño:", round(RUTA_MODELO.stat().st_size / 1024 / 1024, 2), "MB")
    print("SHA-256:", calcular_sha256(RUTA_MODELO))

    print("\n2. ENTORNO DE EJECUCIÓN")
    print("Python:", sys.version)
    print("Sistema:", platform.platform())
    print("Joblib:", joblib.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Pandas:", pd.__version__)

    print("\n3. CARGA DEL MODELO")

    try:
        modelo = joblib.load(RUTA_MODELO)
        print("RESULTADO: modelo cargado correctamente.")
    except Exception as error:
        print("ERROR AL CARGAR EL MODELO:")
        print(repr(error))
        return

    print("\n4. INSPECCIÓN DEL OBJETO")
    inspeccionar_objeto("MODELO PRINCIPAL", modelo)

    # Si el objeto es un Pipeline, inspeccionar sus etapas.
    if hasattr(modelo, "named_steps"):
        print("\n5. ETAPAS DEL PIPELINE")

        for nombre, etapa in modelo.named_steps.items():
            inspeccionar_objeto(nombre, etapa)

    # Si el objeto tiene un estimador interno.
    elif hasattr(modelo, "estimator"):
        print("\n5. ESTIMADOR INTERNO")
        inspeccionar_objeto("ESTIMADOR", modelo.estimator)

    print("\n6. RESULTADO DE AUDITORÍA")
    print("El archivo se encuentra y puede cargarse.")
    print("Esta revisión no demuestra todavía su rendimiento predictivo.")
    print("Tampoco demuestra que las métricas provengan de un test independiente.")

    print("\n" + "=" * 65)
    print("AUDITORÍA FINALIZADA")
    print("=" * 65)


if __name__ == "__main__":
    main()
