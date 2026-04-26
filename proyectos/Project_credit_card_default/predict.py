#!/usr/bin/env python
"""
Script para predecir default de nuevos clientes.
Uso: python predict.py --input nuevos_clientes.csv --output predicciones.csv
"""

import argparse
import pandas as pd
import joblib
import sys
import os

# Agregar src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_engineering import get_feature_columns


def load_models(model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
    """Carga el modelo y el scaler guardados."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Modelo cargado desde {model_path}")
    print(f"Scaler cargado desde {scaler_path}")
    return model, scaler


def predict_batch(model, scaler, df_input, threshold=0.35):
    """Predice para un lote de clientes."""
    from src.data_loader import clean_categorical_variables
    from src.feature_engineering import create_features, scale_features
    
    # Limpiar y crear features
    df_clean = clean_categorical_variables(df_input.copy())
    df_features = create_features(df_clean)
    
    # Seleccionar columnas y escalar
    feature_cols = get_feature_columns()
    X = df_features[feature_cols]
    X_scaled, _ = scale_features(X, scaler=scaler, fit=False)
    
    # Predecir
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    # Crear resultados
    resultados = pd.DataFrame({
        'PROBABILIDAD_DEFAULT': proba,
        'PREDICCION': pred,
        'RIESGO': ['Alto' if p > 0.7 else ('Medio' if p > 0.3 else 'Bajo') for p in proba]
    })
    
    return resultados


def main():
    parser = argparse.ArgumentParser(description='Predecir default de tarjeta de crédito')
    parser.add_argument('--input', '-i', required=True, help='Archivo CSV con datos de clientes')
    parser.add_argument('--output', '-o', default='predicciones.csv', help='Archivo de salida')
    parser.add_argument('--threshold', '-t', type=float, default=0.35, help='Umbral de decisión (0-1)')
    
    args = parser.parse_args()
    
    # Cargar datos
    print(f"Cargando datos desde {args.input}")
    df_input = pd.read_csv(args.input)
    
    # Cargar modelos
    model, scaler = load_models()
    
    # Predecir
    print("Generando predicciones...")
    resultados = predict_batch(model, scaler, df_input, args.threshold)
    
    # Guardar
    resultados.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output}")
    
    # Resumen
    print("\nResumen de predicciones:")
    print(f"   - Total clientes: {len(resultados)}")
    print(f"   - Default predicho: {resultados['PREDICCION'].sum()}")
    print(f"   - Riesgo Alto: {(resultados['RIESGO'] == 'Alto').sum()}")
    print(f"   - Riesgo Medio: {(resultados['RIESGO'] == 'Medio').sum()}")
    print(f"   - Riesgo Bajo: {(resultados['RIESGO'] == 'Bajo').sum()}")


if __name__ == "__main__":
    main()