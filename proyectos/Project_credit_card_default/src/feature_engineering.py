"""
Módulo para crear nuevas features y escalar datos.
Uso: from src.feature_engineering import create_features, scale_features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def create_features(df):
    """
    Crea nuevas variables a partir de los datos originales.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con columnas PAY_1 a PAY_6, BILL_AMT1-6, PAY_AMT1-6
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con nuevas features añadidas
    """
    df = df.copy()
    
    # 1. Máximo atraso en últimos 6 meses
    pagos_atraso = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['MAX_ATRASO'] = df[pagos_atraso].max(axis=1)
    
    # 2. Promedio de atraso
    df['PROMEDIO_ATRASO'] = df[pagos_atraso].mean(axis=1)
    
    # 3. Ratio Pago / Factura por mes
    for i in range(1, 7):
        factura_col = f'BILL_AMT{i}'
        pago_col = f'PAY_AMT{i}'
        ratio_col = f'RATIO_PAGO_{i}'
        
        df[ratio_col] = df.apply(
            lambda row: row[pago_col] / row[factura_col] if row[factura_col] > 0 else 0,
            axis=1
        )
    
    # 4. Ratio Pago Total / Factura Total
    facturas_totales = df[[f'BILL_AMT{i}' for i in range(1,7)]].sum(axis=1)
    pagos_totales = df[[f'PAY_AMT{i}' for i in range(1,7)]].sum(axis=1)
    
    df['RATIO_PAGO_TOTAL'] = pagos_totales / facturas_totales.replace(0, np.nan)
    df['RATIO_PAGO_TOTAL'] = df['RATIO_PAGO_TOTAL'].fillna(0)
    
    return df


def get_feature_columns():
    """
    Retorna la lista de columnas a usar en el modelo.
    
    Returns:
    --------
    list
        Nombres de las features
    """
    return [
        'LIMITE_CREDITO', 'SEXO', 'EDUCACION', 'ESTADO_CIVIL', 'EDAD',
        'MAX_ATRASO', 'PROMEDIO_ATRASO', 'RATIO_PAGO_TOTAL'
    ] + [f'RATIO_PAGO_{i}' for i in range(1,7)]


def scale_features(df, scaler=None, fit=False, save_path='../models/scaler.pkl'):
    """
    Escala las features numéricas usando StandardScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con las features
    scaler : StandardScaler, optional
        Scaler pre-entrenado (si fit=False)
    fit : bool
        Si True, entrena un nuevo scaler y lo guarda
    save_path : str
        Ruta para guardar el scaler
    
    Returns:
    --------
    tuple (df_scaled, scaler)
    """
    df_scaled = df.copy()
    feature_cols = get_feature_columns()
    
    if fit:
        scaler = StandardScaler()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, save_path)
        print(f"Scaler guardado en {save_path}")
    else:
        if scaler is None:
            raise ValueError("Debe proporcionar un scaler cuando fit=False")
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    return df_scaled, scaler


if __name__ == "__main__":
    # Prueba rápida
    from src.data_loader import load_and_clean_data, clean_categorical_variables
    
    df = load_and_clean_data()
    df = clean_categorical_variables(df)
    df = create_features(df)
    
    print("Features creadas:", [col for col in df.columns if 'RATIO' in col or 'MAX' in col])
    print(f"\nDimensiones finales: {df.shape}")