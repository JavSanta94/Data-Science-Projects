"""
Módulo para entrenar y guardar el modelo final.
Uso: from src.model_training import train_best_model, predict_default
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Parámetros óptimos encontrados en GridSearchCV
BEST_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 7,
    'n_estimators': 200,
    'subsample': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Umbral óptimo (se ajustará después)
OPTIMAL_THRESHOLD = 0.35


def train_best_model(X, y, save_path='../models/best_model.pkl', test_size=0.2):
    """
    Entrena el modelo XGBoost optimizado y lo guarda.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target (DEFAULT)
    save_path : str
        Ruta para guardar el modelo
    test_size : float
        Proporción para test
    
    Returns:
    --------
    tuple (model, X_test, y_test, y_proba)
    """
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Calcular scale_pos_weight
    scale = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Crear modelo con parámetros óptimos
    model = xgb.XGBClassifier(
        **BEST_PARAMS,
        scale_pos_weight=scale
    )
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Evaluar
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Modelo entrenado y guardado en {save_path}")
    print(f"AUC en test: {auc:.4f}")
    
    # Guardar modelo
    joblib.dump(model, save_path)
    
    return model, X_test, y_test, y_proba


def predict_default(model, scaler, df_new, threshold=OPTIMAL_THRESHOLD):
    """
    Predice probabilidad de default para nuevos clientes.
    
    Parameters:
    -----------
    model : objeto sklearn
        Modelo entrenado
    scaler : StandardScaler
        Scaler entrenado
    df_new : pd.DataFrame
        Datos de nuevos clientes (mismo formato que original)
    threshold : float
        Umbral de decisión (default 0.35)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con predicciones y probabilidades
    """
    from src.data_loader import clean_categorical_variables
    from src.feature_engineering import create_features, scale_features, get_feature_columns
    
    # Limpiar y crear features
    df_clean = clean_categorical_variables(df_new.copy())
    df_features = create_features(df_clean)
    
    # Seleccionar columnas y escalar
    feature_cols = get_feature_columns()
    X = df_features[feature_cols]
    X_scaled, _ = scale_features(X, scaler=scaler, fit=False)
    
    # Predecir
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    # Resultados
    resultados = df_new[['ID']] if 'ID' in df_new.columns else pd.DataFrame(index=df_new.index)
    resultados['PROBABILIDAD_DEFAULT'] = proba
    resultados['PREDICCION'] = pred
    resultados['RIESGO'] = resultados['PROBABILIDAD_DEFAULT'].apply(
        lambda x: 'Alto' if x > 0.7 else ('Medio' if x > 0.3 else 'Bajo')
    )
    
    return resultados


if __name__ == "__main__":
    # Prueba rápida
    from src.data_loader import load_and_clean_data, clean_categorical_variables
    from src.feature_engineering import create_features, get_feature_columns, scale_features
    
    # Cargar y preparar datos
    df = load_and_clean_data()
    df = clean_categorical_variables(df)
    df = create_features(df)
    
    # Seleccionar features
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df['DEFAULT']
    
    # Entrenar modelo
    model, X_test, y_test, y_proba = train_best_model(X, y)
    
    print("\nPrueba completada")