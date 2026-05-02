# Challenge: Modelo Predictivo de Incumplimiento Crediticio

Análisis técnico completo de construcción de un modelo de machine learning para predecir incumplimiento crediticio a 60 días, incluyendo evaluación de modelos de proveedores externos.

## 📋 Descripción General

Este proyecto implementa un flujo de ciencia de datos end-to-end para una institución financiera (Solventa), desarrollando un modelo predictivo de mora sobre un portfolio de clientes de alto riesgo y evaluando soluciones comerciales de dos proveedores de scoring crediticio.

**Período de datos**: Enero 2017 — Septiembre 2018  
**Total de registros**: 4,315 clientes  
**Target**: Mora60 (incumplimiento a 60 días)  
**Tasa de default**: 11.7%

## 🎯 Resultados Principales

| Métrica | Valor | Interpretación |
|---------|-------|---|
| **AUC-ROC** | 0.6681 | Discriminación moderada-buena sin bureau externo |
| **KS** | 0.3293 | Estándar operativo para originación (> 0.20) |
| **Gini** | 0.3363 | Capacidad discriminante comprobada |
| **Modelo** | Random Forest | Ganador sobre LR, XGB, LightGBM |
| **Umbral** | 0.49 | Balanceado: 63.5% aprobados, 6.2% malos aprobados |

## 📁 Estructura del Repositorio

```
├── README.md                              # Este archivo
│
├── notebooks/
│   ├── 01_limpieza_datos.py              # Script de limpieza: 6 correcciones aplicadas
│   ├── 02_EDA.ipynb                      # Análisis exploratorio + IV de todas las variables
│   ├── 03_feature_engineering.ipynb      # WoE encoding, log1p, ratios, agrupaciones
│   ├── 04_modelado.ipynb                 # Pasos 1-10: entrenamiento, evaluación, scorecard
│
├── datos/
│   └── ProductoNuevo.xlsx                # Dataset de entrada (4,315 registros, 22 variables)
│
└── artefactos/
    ├── train_model.parquet               # Train set procesado (3,452 clientes, 8 features)
    ├── test_model.parquet                # Test set procesado (863 clientes, 8 features)
    └── fe_fit_params.pkl                 # Parámetros WoE y bins aprendidos en train
```

## 🔍 Hallazgos Clave

### 1. Arquitectura de Segmentación (v2)
- **41% sin mora previa**: MoraMax=0 → Aprobación automática (100% pagan)
- **59% con mora previa**: MoraMax>0 → Aplicar modelo RF (AUC esperado > 0.80)

### 2. Variables Más Relevantes
1. **Antigüedad en sistema** (28%) — Clientes sin historial son 2.3x más riesgosos
2. **Antigüedad como cliente** (22%) — Relación previa reduce riesgo
3. **Endeudamiento (PORCEND)** (18%) — Sobreendeudados son más riesgosos
4. **Nivel educativo** (10%) — Mayor formación correlaciona con menor mora
5. **Edad** (9%) — Mayor estabilidad relativa con edad

### 3. Capacidad Discriminante por Decil
- **D1** (20% de mayor riesgo): 27.6% de tasa de malo → Lift 2.58x
- **D10** (20% de menor riesgo): 0% de tasa de malo → Separación perfecta
- **Rechazando D1+D2**: Se captura el 42.4% de los malos aprobando el 80%

## 🛠️ Metodología

### Tratamiento de Datos
- **7 correcciones** aplicadas sin eliminar registros (0 registros perdidos)
- **Outliers**: Estrategia conservadora — Random Forest es robusto por naturaleza
- **Imputación**: Median-based para GastoArriendo imputados, con flag de imputación

### Feature Engineering
- **log1p** en variables sesgadas (TiempoClienteMeses, Tiempo_SistemaFro)
- **WoE Encoding** para categorías (aprendido en train, frozen en test)
- **Ratios personalizados**: capacidad_pago, carga_total
- **Agrupación inteligente**: Fusión de categorías poco representadas

### División Train/Test
- **Temporal 80/20** (no aleatorio)
- **Justificación**: Tendencia bajista confirmada (mora -0.28pp/mes, p=0.046)
- **Train**: 3,452 clientes (11.9% default) — Enero 2017 a Julio 2018
- **Test**: 863 clientes (10.7% default) — Julio a Septiembre 2018

### Modelado y Validación
- **4 modelos evaluados**: LR, Random Forest, XGBoost, LightGBM
- **Balanceo**: class_weight='balanced' (desbalance moderado 7.4:1)
- **CV**: TimeSeriesSplit (4 folds) — Respeta orden temporal
- **Búsqueda**: RandomizedSearchCV (30 iteraciones, métrica: AUC-ROC)
- **Sin sobreajuste**: Gap CV-Test < 0.05, metricas consistentes en periodo reciente

## 📊 Entregables

### Notebooks Reproducibles
- Limpieza de datos con 6 correcciones documentadas
- EDA con cálculo de IV para cada variable
- Feature Engineering con transformaciones paso a paso
- Modelado con pasos 1-10 (entrenamiento, evaluación, scorecard, importancia)
- Evaluación de proveedores con métricas AUC, Gini, KS, PSI

## 🚀 Cómo Reproducir

### Requisitos
```bash
python >= 3.8
pandas >= 1.3
scikit-learn >= 1.0
matplotlib >= 3.4
xgboost >= 1.5
lightgbm >= 3.2
```

### Instalación
```bash
git clone https://github.com/[usuario]/challenge-modelo-incumplimiento
cd challenge-modelo-incumplimiento
pip install -r requirements.txt
```

### Ejecución
```bash
# 1. Limpieza de datos
python notebooks/01_limpieza_datos.py

# 2. Ejecutar notebooks en orden
jupyter notebook notebooks/02_EDA.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_modelado.ipynb
```

## 📈 Métricas de Desempeño

### Test Set (863 clientes)
```
Random Forest (modelo seleccionado):
  Precision:   18.4%  (de los rechazados, 18.4% son malos reales)
  Recall:      63.0%  (captura el 63% de los malos reales)
  F1-Score:    0.286  (equilibrio Precision-Recall)
  AUC-ROC:     0.668  (discriminación comprobada)
  KS:          0.329  (estándar operativo)
```

### Comparativa de Modelos
| Modelo | AUC | KS | Gini |
|--------|-----|-----|------|
| **Random Forest** | **0.6681** | **0.3293** | **0.3363** |
| Logistic Regression | 0.6396 | 0.2744 | 0.2793 |
| XGBoost | 0.5742 | 0.1785 | 0.1484 |
| LightGBM | 0.5976 | 0.1796 | 0.1953 |

## ⚠️ Limitaciones Documentadas

1. **Señal limitada**: IV máximo 0.103 sin variables de bureau externo
2. **Población de malos pequeña**: 412 en train, 92 en test
3. **Tendencia temporal**: Mora bajó de 12.6% a 7% (2017-2018)
4. **GastosFamiliares como proxy**: 96.9% calculado automáticamente
5. **Estacionalidad no modelada**: Mora varía significativamente por mes

## 🎯 Próximos Pasos (Prioridad)

| Nivel | Acción | Impacto |
|-------|--------|--------|
| 🔴 ALTA | Arquitectura segmentada con MoraMax | AUC > 0.80 |
| 🔴 ALTA | Integración variables bureau externo | +0.05 a +0.10 AUC |
| 🟡 MEDIA | Variable estacionalidad mensual | Mejor estabilidad temporal |
| 🟡 MEDIA | Monitoreo PSI, KS mensual | Gobierno del modelo |
| 🟢 BAJA | SHAP values para interpretabilidad | Explicabilidad regulatoria |

## 📚 Referencias Técnicas

- **Information Value (IV)**: Criterio de pre-selección de features estándar en modelos crediticios
- **WoE Encoding**: Transforma categorías preservando relación con target, sin expandir dimensionalidad
- **TimeSeriesSplit**: Validación cruzada que respeta orden temporal, previene data leakage
- **KS (Kolmogorov-Smirnov)**: Métrica operativa principal en originación crediticia (máxima separación entre buenos/malos)
- **Gini**: Métrica de desigualdad equivalente a 2×AUC − 1

## 👤 Autor

Data Scientist  Mayo 2026

## 📄 Licencia

MIT License

---
