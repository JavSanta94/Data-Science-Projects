# Credit Card Default Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-orange)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green)](https://scikit-learn.org/)

> Predicción de morosidad en tarjetas de crédito usando Machine Learning

## 📌 Problema de Negocio

Un banco o entidad financiera emite tarjetas de crédito. Cada mes, algunos clientes no pagan su factura mínima, lo que genera pérdidas por morosidad. El banco quiere reducir este riesgo.

**Problema a resolver:** Predecir si un cliente caerá en mora (default) en el pago de su tarjeta de crédito el próximo mes, basándose en su historial de pagos, facturación y datos demográficos.

## ❓ Pregunta a Responder

Dado el comportamiento histórico de un cliente (demografía, límite de crédito, historial de pagos atrasados, montos facturados y pagados en los últimos 6 meses), **¿cuál es la probabilidad de que no pague su tarjeta el próximo mes?**

**Preguntas secundarias:**
- ¿Qué factores están más correlacionados con el impago?
- ¿Los clientes con mayor límite de crédito son más propensos a pagar o a caer en mora?
- ¿El historial de pagos atrasados es el predictor más fuerte?
- ¿Existen diferencias significativas por sexo, nivel educativo o estado civil?

## 🎯 Resultado Esperado por el Negocio

- Modelo de clasificación que asigne **probabilidad de default** a cada cliente
- **AUC-ROC > 0.75** (superar baseline)
- **Recall alto** para detectar morosos (es más costoso no detectarlos)

## 📊 Dataset

**Fuente:** UCI Credit Card Dataset  
**Registros:** 30,000 clientes  
**Features:** 25 variables (demografía, facturación, pagos, historial de atrasos)

| Variable | Descripción |
|----------|-------------|
| LIMITE_CREDITO | Límite de crédito asignado |
| SEXO | 1=Hombre, 2=Mujer |
| EDUCACION | 1=Posgrado, 2=Universidad, 3=Secundaria, 4=Otros |
| ESTADO_CIVIL | 1=Casado, 2=Soltero, 3=Otros |
| EDAD | Edad en años |
| PAY_1 a PAY_6 | Historial de atraso (meses recientes) |
| BILL_AMT1 a BILL_AMT6 | Monto facturado por mes |
| PAY_AMT1 a PAY_AMT6 | Monto pagado por mes |
| DEFAULT | Variable objetivo (1=default, 0=no default) |

## 🏗️ Estructura del Proyecto

credit_card_default/
│
├── data/
│ └── UCI_Credit_Card.csv
│
├── notebooks/
│ ├── 01_EDA_and_Cleaning.ipynb # Análisis exploratorio
│ ├── 02_Feature_Engineering.ipynb # Creación de variables
│ └── 03_Modeling_and_Evaluation.ipynb # Modelado y evaluación
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Carga y limpieza
│ ├── feature_engineering.py # Transformaciones
│ └── model_training.py # Entrenamiento y predicción
│
├── models/
│ ├── best_model.pkl # Modelo final
│ ├── scaler.pkl # Escalador
│
├── predict.py # Script de predicción
├── requirements.txt
└── README.md

## 🚀 Modelos Probados

| Modelo | AUC (test) | AP |
|--------|------------|-----|
| Regresión Logística | 0.7090 | 0.4544 |
| Regresión Logística Balanceada | 0.7087 | 0.4432 |
| Random Forest | 0.7387 | 0.4943 |
| Random Forest Balanceado | 0.7375 | 0.4932 |
| XGBoost (default) | 0.7409 | 0.4864 |
| **XGBoost Optimizado** | **0.7566** | **0.5195** |

## ✅ Mejor Modelo

**XGBoost optimizado con GridSearchCV**

### Hiperparámetros óptimos:
```python
{
    'learning_rate': 0.01,
    'max_depth': 7,
    'n_estimators': 200,
    'subsample': 0.8,
    'scale_pos_weight': 3.52  # balanceo de clases
}

Rendimiento:
Métrica	Valor
AUC-ROC (validación)	0.7748
AUC-ROC (test)	0.7566 ✅
Average Precision	0.5195

Features más importantes:
Feature	Importancia
MAX_ATRASO	54.7%
PROMEDIO_ATRASO	7.2%
RATIO_PAGO_TOTAL	3.8%
RATIO_PAGO_3	3.6%
LIMITE_CREDITO	3.6%

🎯 Recomendaciones de Negocio
Recomendación	Justificación
Threshold = 0.35	Balance entre Recall y Precisión
Monitorear mensualmente	El comportamiento de pago cambia en el tiempo
Alertar clientes con MAX_ATRASO ≥ 2	Alto riesgo de default
Ofrecer plan de pagos anticipado	Antes de que caigan en mora