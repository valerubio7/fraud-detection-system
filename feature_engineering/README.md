# Feature Engineering

Calcula las features que alimentan al modelo de detección de fraude. Tiene dos variantes: **offline** (batch, para entrenamiento) y **online** (streaming, para inferencia en producción).

## Estructura

```
feature_engineering/
└── offline/       # Implementado — replica en batch la lógica de streaming
```

## Offline (`offline/`)

Usado durante entrenamiento. Procesa transacciones históricas agrupadas por `user_id`, ordenadas por timestamp, garantizando zero leakage temporal (cada fila solo ve transacciones estrictamente anteriores).

### `featurizer.py` — `TransactionFeaturizer`

API estilo scikit-learn: `fit()` / `transform()` / `fit_transform()`. Computa **18 features** por transacción:

| Categoría | Features |
|---|---|
| **Directas** (6) | `log_amount`, `hour_of_day`, `day_of_week`, `merchant_category_encoded`, `country_encoded`, `device_type_encoded` |
| **Ventana temporal** (7) | `tx_count_1h`, `tx_count_24h`, `tx_count_7d`, `amount_sum_1h`, `amount_sum_24h`, `tx_velocity_1h`, `seconds_since_last_tx` |
| **Perfil histórico** (5) | `amount_ratio_vs_user_avg`, `is_country_new`, `distinct_countries_seen`, `is_merchant_new`, `distinct_merchants_seen` |

```python
from feature_engineering.offline.featurizer import TransactionFeaturizer

featurizer = TransactionFeaturizer()
featurizer.fit(X_train, y_train)
X_train_feat = featurizer.transform(X_train)
X_test_feat = featurizer.transform(X_test)

# Opcional: reducir a features seleccionadas
featurizer.apply_selection(selection_report)
```

Columnas requeridas de entrada: `transaction_id`, `user_id`, `merchant_id`, `merchant_category`, `amount`, `country`, `device_type`, `timestamp`, `is_fraud`.

### `encoders.py` — `CategoricalEncoderPipeline`

Pipeline de encoding categórico serializable via joblib:

| Columna | Encoder | Motivo |
|---|---|---|
| `merchant_category` | `TargetEncoder` | Alta cardinalidad, encoding supervisado con suavizado |
| `country` | `TargetEncoder` | Alta cardinalidad, encoding supervisado con suavizado |
| `device_type` | `OrdinalEncoder` | Baja cardinalidad (3 valores), encoding ordinal |

```python
from feature_engineering.offline.encoders import CategoricalEncoderPipeline

pipeline = CategoricalEncoderPipeline()
pipeline.fit(X_train, y_train)
X_encoded = pipeline.transform(X_train)
pipeline.save("encoders.joblib")
```

### `feature_selection.py` — Feature selection

Reduce las 18 features eliminando las de baja importancia o redundantes:

| Función | Descripción |
|---|---|
| `compute_xgboost_importance(X, y)` | Importancia gain con XGBoost |
| `compute_correlation_matrix(X)` | Matriz de correlación de Pearson |
| `find_redundant_features(X, threshold=0.85)` | Pares con correlación > threshold |
| `select_features(X, y, ...)` | Orquestador: importancia → eliminar baja → eliminar redundante → Boruta |

```python
from feature_engineering.offline.feature_selection import select_features

report = select_features(X_train, y_train)
print(report.selected_features)      # features que se quedan
print(report.drop_reason)            # por qué se descartó cada una
```

### `class_imbalance.py` — Manejo de desbalance

Compara dos estrategias para el desbalance ~49:1 típico en fraude:

- **SMOTE** — oversampling sintético de la clase minoritaria
- **scale_pos_weight** — weighting interno de XGBoost

```python
from feature_engineering.offline.class_imbalance import run_imbalance_analysis

report = run_imbalance_analysis(X_train, y_train, X_val, y_val)
print(report.recommended_strategy)   # la que mejor F1 obtuvo
```

## Flujo de uso en entrenamiento

```
Raw CSV → TransactionFeaturizer → 18 features
                ↓
         Feature selection  →  features reducidas
                ↓
         Imbalance analysis →  estrategia elegida
                ↓
         Entrenamiento del modelo
```
