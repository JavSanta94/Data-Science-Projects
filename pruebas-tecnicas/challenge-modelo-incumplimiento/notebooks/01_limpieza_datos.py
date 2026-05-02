"""
=============================================================================
PRUEBA TÉCNICA - DATA SCIENTIST JR. | SOLVENTA
=============================================================================
Script 01 — Limpieza y calidad de datos
Autor   : Javier Yesid Ladino Santamaría
Fecha   : 2026-04

Descripción:
    Aplica correcciones sobre errores de captura e inconsistencias detectadas
    durante la revisión de calidad de datos del dataset ProductoNuevo.

Hallazgos que motivan cada corrección:
    1. GastoArriendo = 0 en vivienda ALQUILADA  → dato faltante enmascarado
    2. GastoArriendo = 1                         → placeholder / error de captura
    3. GastoArriendo > 0 en vivienda PROPIA      → inconsistencia lógica
    4. TiempoActividadAnios = 866880             → outlier extremo (error de digitación)
    5. TiempoActividadAnios = 72 (x2 registros)  → valor fuera del rango esperado (p99=20)
    6. GastosFamiliares ≈ 0.67 * Ingresos        → relación mecánica, multicolinealidad r=0.96
       → se conserva GastosFamiliares y se elimina la redundancia vía feature engineering
    7. PORCEND > 1                               → documentado, NO se corrige (18.6% registros,
                                                   puede representar sobreendeudamiento real)
=============================================================================
"""

import pandas as pd
import numpy as np


# =============================================================================
# CONSTANTES
# =============================================================================

# Percentil 99 de TiempoActividadAnios calculado sobre la distribución limpia
CAP_TIEMPO_ACTIVIDAD = 20.0

# Mediana de GastoArriendo calculada únicamente sobre registros ALQUILADA con valor > 1
# (excluye ceros y el placeholder=1 para no contaminar la imputación)
MEDIANA_ARRIENDO_ALQUILADA = 150_000.0


# =============================================================================
# FUNCIÓN PRINCIPAL DE LIMPIEZA
# =============================================================================

def limpiar_datos(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aplica correcciones de calidad sobre el dataset ProductoNuevo.

    Parámetros
    ----------
    df      : DataFrame original cargado desde ProductoNuevo.xlsx
    verbose : Si True, imprime un resumen de registros afectados por cada corrección

    Retorna
    -------
    df_clean : DataFrame corregido con columnas de flags adicionales
    reporte  : dict con conteo de registros afectados por cada regla
    """

    df_clean = df.copy()
    reporte = {}

    # -------------------------------------------------------------------------
    # REGLA 1 — GastoArriendo = 1 → corregir a 0
    # -------------------------------------------------------------------------
    # Valor económicamente imposible. Aparece en registros FAMILIAR,
    # consistente con no pagar arriendo. Se interpreta como placeholder.
    # -------------------------------------------------------------------------
    mask_placeholder = df_clean["GastoArriendo"] == 1
    reporte["gasto_arriendo_placeholder_1"] = mask_placeholder.sum()
    df_clean.loc[mask_placeholder, "GastoArriendo"] = 0

    # -------------------------------------------------------------------------
    # REGLA 2 — GastoArriendo > 0 en Tipo_Vivienda = PROPIA → corregir a 0
    # -------------------------------------------------------------------------
    # Un cliente con vivienda propia no debería tener gasto de arriendo.
    # Solo 1 registro afectado → error de digitación.
    # -------------------------------------------------------------------------
    mask_propia_arriendo = (
        (df_clean["Tipo_Vivienda"] == "PROPIA") & (df_clean["GastoArriendo"] > 0)
    )
    reporte["propia_con_arriendo"] = mask_propia_arriendo.sum()
    df_clean.loc[mask_propia_arriendo, "GastoArriendo"] = 0

    # -------------------------------------------------------------------------
    # REGLA 3 — GastoArriendo = 0 en Tipo_Vivienda = ALQUILADA → imputar
    # -------------------------------------------------------------------------
    # 84 registros (33.7% de ALQUILADA) reportan $0 de arriendo.
    # Se imputa con la mediana de arriendos válidos del segmento ALQUILADA.
    # Se crea un flag binario para que el modelo pueda aprender de esta
    # imputación sin tomarla como un valor observado real.
    # -------------------------------------------------------------------------
    mask_alquilada_cero = (
        (df_clean["Tipo_Vivienda"] == "ALQUILADA") & (df_clean["GastoArriendo"] == 0)
    )
    reporte["alquilada_arriendo_cero"] = mask_alquilada_cero.sum()

    # Flag ANTES de imputar
    df_clean["flag_arriendo_imputado"] = mask_alquilada_cero.astype(int)
    df_clean.loc[mask_alquilada_cero, "GastoArriendo"] = MEDIANA_ARRIENDO_ALQUILADA

    # -------------------------------------------------------------------------
    # REGLA 4 — TiempoActividadAnios = 866,880 → outlier crítico
    # -------------------------------------------------------------------------
    # Un solo registro con valor absurdo (posiblemente minutos ingresados como años).
    # Se capea al p99 de la distribución limpia (20 años).
    # -------------------------------------------------------------------------
    mask_outlier_extremo = df_clean["TiempoActividadAnios"] == 866_880
    reporte["tiempo_actividad_outlier_extremo"] = mask_outlier_extremo.sum()
    df_clean.loc[mask_outlier_extremo, "TiempoActividadAnios"] = CAP_TIEMPO_ACTIVIDAD

    # -------------------------------------------------------------------------
    # REGLA 5 — TiempoActividadAnios = 72 → fuera del rango esperado
    # -------------------------------------------------------------------------
    # Dos registros con 72 años de actividad laboral. El p99 es 20 años.
    # Para un empleado activo esto es biológicamente improbable.
    # Se capea al mismo techo de 20 años.
    # -------------------------------------------------------------------------
    mask_outlier_72 = df_clean["TiempoActividadAnios"] > CAP_TIEMPO_ACTIVIDAD
    reporte["tiempo_actividad_capeado_p99"] = mask_outlier_72.sum()
    df_clean.loc[mask_outlier_72, "TiempoActividadAnios"] = CAP_TIEMPO_ACTIVIDAD

    # -------------------------------------------------------------------------
    # REGLA 6 — TiempoActividadAnios = 0 con TiempoClienteMeses > 12
    # -------------------------------------------------------------------------
    # 66 clientes llevan más de un año en la entidad pero reportan 0 años
    # de actividad laboral. Se crea un flag para que el modelo identifique
    # esta inconsistencia sin modificar el valor (puede ser dato faltante
    # o un trabajador informal sin registro formal de actividad).
    # -------------------------------------------------------------------------
    mask_actividad_cero_cliente_antiguo = (
        (df_clean["TiempoActividadAnios"] == 0) & (df_clean["TiempoClienteMeses"] > 12)
    )
    reporte["actividad_cero_cliente_antiguo"] = mask_actividad_cero_cliente_antiguo.sum()
    df_clean["flag_actividad_inconsistente"] = mask_actividad_cero_cliente_antiguo.astype(int)

    # -------------------------------------------------------------------------
    # NOTA — GastosFamiliares vs Ingresos (r = 0.964)
    # -------------------------------------------------------------------------
    # El 75% de los registros tiene GastosFamiliares = exactamente 67% de Ingresos,
    # lo que indica un cálculo automático en el sistema de originación.
    # NO se elimina ninguna variable aquí; la multicolinealidad se manejará
    # en el Feature Engineering mediante:
    #   - Ratio capacidad_pago = (Ingresos - GastosFamiliares) / Ingresos
    #   - Descarte de una de las dos variables en la selección final
    # -------------------------------------------------------------------------
    reporte["gasto_ingreso_colineales_documentado"] = (
        (df_clean["GastosFamiliares"] / df_clean["Ingresos"]).round(3) == 0.670
    ).sum()

    # -------------------------------------------------------------------------
    # NOTA — PORCEND > 1 (18.6% de registros)
    # -------------------------------------------------------------------------
    # Se documenta pero NO se corrige. El porcentaje de endeudamiento puede
    # superar el 100% en clientes sobreendeudados, que es precisamente el
    # segmento de alto riesgo que este producto busca evaluar.
    # -------------------------------------------------------------------------
    reporte["porcend_mayor_1_documentado"] = (df_clean["PORCEND"] > 1).sum()

    # -------------------------------------------------------------------------
    # REPORTE FINAL
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("REPORTE DE LIMPIEZA — ProductoNuevo")
        print("=" * 60)
        print(f"  Registros originales           : {len(df):,}")
        print(f"  Registros después de limpieza  : {len(df_clean):,}")
        print()
        print("  Correcciones aplicadas:")
        print(f"    GastoArriendo = 1 (placeholder)       : {reporte['gasto_arriendo_placeholder_1']:>4} registros → 0")
        print(f"    PROPIA con arriendo > 0                : {reporte['propia_con_arriendo']:>4} registros → 0")
        print(f"    ALQUILADA con arriendo = 0 (imputado)  : {reporte['alquilada_arriendo_cero']:>4} registros → $150,000")
        print(f"    TiempoActividad = 866,880 (outlier)    : {reporte['tiempo_actividad_outlier_extremo']:>4} registro  → 20")
        print(f"    TiempoActividad > 20 (cap p99)         : {reporte['tiempo_actividad_capeado_p99']:>4} registros → 20")
        print()
        print("  Flags creados:")
        print(f"    flag_arriendo_imputado                 : {df_clean['flag_arriendo_imputado'].sum():>4} registros con valor 1")
        print(f"    flag_actividad_inconsistente           : {df_clean['flag_actividad_inconsistente'].sum():>4} registros con valor 1")
        print()
        print("  Documentados (sin corrección):")
        print(f"    GastosFamiliares ≈ 0.67 * Ingresos    : {reporte['gasto_ingreso_colineales_documentado']:,} registros (manejo en FE)")
        print(f"    PORCEND > 1                            : {reporte['porcend_mayor_1_documentado']:,} registros (sobreendeudamiento válido)")
        print("=" * 60)

    return df_clean, reporte


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    # Carga
    df_raw = pd.read_excel("ProductoNuevo.xlsx")

    # Limpieza
    df_clean, reporte = limpiar_datos(df_raw, verbose=True)

    # Validación rápida post-limpieza
    print("\nVALIDACIÓN POST-LIMPIEZA")
    print("-" * 40)
    print(f"GastoArriendo = 1 restantes       : {(df_clean['GastoArriendo'] == 1).sum()}")
    print(f"ALQUILADA con arriendo = 0         : {((df_clean['Tipo_Vivienda']=='ALQUILADA') & (df_clean['GastoArriendo']==0)).sum()}")
    print(f"PROPIA con arriendo > 0            : {((df_clean['Tipo_Vivienda']=='PROPIA') & (df_clean['GastoArriendo']>0)).sum()}")
    print(f"TiempoActividad > 20               : {(df_clean['TiempoActividadAnios'] > 20).sum()}")
    print(f"Columnas totales (con flags)       : {df_clean.shape[1]}")