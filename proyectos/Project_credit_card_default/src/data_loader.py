"""
Módulo para cargar y limpiar datos iniciales.
Uso: from src.data_loader import load_and_clean_data
"""

import pandas as pd
import numpy as np

def load_and_clean_data(filepath='UCI_Credit_Card.csv'):
    """
    Carga el dataset UCI Credit Card y aplica limpieza inicial.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV
    
    Returns:
    --------
    pd.DataFrame
        DataFrame limpio con columnas renombradas
    """
    # Cargar datos
    df = pd.read_csv(filepath)
    
    # Eliminar ID (no sirve para predecir)
    df = df.drop('ID', axis=1)
    
    # Renombrar columnas para legibilidad
    df.rename(columns={
        'default.payment.next.month': 'DEFAULT',
        'PAY_0': 'PAY_1',
        'LIMIT_BAL': 'LIMITE_CREDITO',
        'SEX': 'SEXO',
        'EDUCATION': 'EDUCACION',
        'MARRIAGE': 'ESTADO_CIVIL',
        'AGE': 'EDAD'
    }, inplace=True)
    
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_categorical_variables(df):
    """
    Limpia variables categóricas combinando valores raros.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con columnas 'EDUCACION' y 'ESTADO_CIVIL'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con categorías limpias
    """
    # EDUCACION: 1,2,3 se mantienen, el resto (0,5,6) → 4 (Otros)
    df['EDUCACION'] = df['EDUCACION'].apply(lambda x: x if x in [1,2,3] else 4)
    
    # ESTADO_CIVIL: 1,2 se mantienen, 0 y 3 → 3 (Otros)
    df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].apply(lambda x: x if x in [1,2] else 3)
    
    return df


if __name__ == "__main__":
    # Prueba rápida
    df = load_and_clean_data()
    df = clean_categorical_variables(df)
    print("\nPrimeras 5 filas:")
    print(df.head())