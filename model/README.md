# Model

Entrena, evalúa y promociona el modelo XGBoost de detección de fraude. Todo el ciclo está integrado con MLflow para tracking de experimentos y registry de modelos.

## Pipeline de entrenamiento

```
TimescaleDB → TransactionFeaturizer (18 features) → Feature Selection (15 features)
                                                         ↓
                                              Temporal Split (70/15/15)
                                                         ↓
                                              Optuna (opcional) → params
                                                         ↓
                                              XGBoost train con early stopping
                                                         ↓
                                              Optimal threshold (cost-aware)
                                                         ↓
                                              Evaluación en test set
                                                         ↓
                                              MLflow log + register (Staging)
```

## Scripts

### `train.py` — Entry point de entrenamiento

```bash
docker compose run --rm model python -m model.train [flags]
```

| Flag | Default | Descripción |
|---|---|---|
| `--seed` | `42` | Random seed |
| `--output-dir` | `artifacts/model` | Directorio de salida |
| `--limit` | — | Limitar filas cargadas de TimescaleDB |
| `--no-mlflow` | — | Deshabilitar MLflow |
| `--tune` | — | Ejecutar Optuna hyperparameter tuning |
| `--n-trials` | `30` | Trials de Optuna |
| `--cost-fn` | `100.0` | Costo de un falso negativo (fraude no detectado) |
| `--cost-fp` | `5.0` | Costo de un falso positivo (legítima bloqueada) |
| `--threshold` | — | Umbral fijo (si no se pasa, se calcula el óptimo) |

Carga datos desde TimescaleDB, construye features con `TransactionFeaturizer`, valida que las features seleccionadas coincidan con `selected_features.py`, hace split temporal (70/15/15), entrena XGBoost con early stopping, calcula threshold óptimo minimizando costo, evalúa en test, logea todo en MLflow y registra el modelo en **Staging**.

### `tuning.py` — Optuna hyperparameter search

Optimiza 9 hiperparámetros maximizando PR-AUC en validación (métrica robusta para clases desbalanceadas):

| Parámetro | Rango de búsqueda |
|---|---|
| `n_estimators` | 100 – 600 |
| `max_depth` | 3 – 9 |
| `learning_rate` | 0.01 – 0.3 (log) |
| `min_child_weight` | 1 – 10 |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `gamma` | 0.0 – 5.0 |
| `reg_alpha` | 1e-4 – 10.0 (log) |
| `reg_lambda` | 1e-4 – 10.0 (log) |
| `scale_pos_weight` | 0.5x – 2.0x del weight calculado |

Cada trial se logea como nested run de MLflow.

### `evaluate.py` — Quality gates + champion comparison

Evalúa un modelo registrado contra thresholds de calidad:

```bash
docker compose run --rm model python -m model.evaluate \
    --model-name fraud-detection-model \
    --model-version 3 \
    --compare
```

| Gate | Threshold |
|---|---|
| F1-score (fraud) | >= 0.85 |
| AUC-ROC | >= 0.90 |
| Latency P99 | <= 50 ms |

Con `--compare` compara contra el champion en Production: el challenger gana si supera su F1 por al menos 0.02.

### `promote.py` — Staging → Production

```bash
docker compose run --rm model python -m model.promote \
    --model-name fraud-detection-model \
    --model-version 3
```

Verifica que el modelo esté en **Staging**, registra el deployment en PostgreSQL (`model_deployments`), ejecuta `activate_model_version()` y mueve el modelo a **Production** en MLflow (archivando la versión anterior).

## Módulos de soporte

### `metrics.py`

- `evaluate_model()` — métricas completas: precision, recall, F1, ROC-AUC, PR-AUC, matriz de confusión, costos, fraud_detected_pct
- `find_optimal_threshold()` — busca el threshold que minimiza `FN * cost_fn + FP * cost_fp`
- `compute_threshold_metrics()` — precision, recall, F1 para cada threshold

### `plots.py`

Genera 5 gráficos durante el entrenamiento:

| Archivo | Descripción |
|---|---|
| `confusion_matrix.png` | Matriz de confusión (legítima vs fraude) |
| `roc_curve.png` | Curva ROC con AUC |
| `pr_curve.png` | Precision-Recall curve con línea base |
| `feature_importance.png` | Top features por gain |
| `threshold_analysis.png` | F1/precision/recall vs threshold |

### `selected_features.py`

Lista definitiva de **15 features** seleccionadas (de 18 originales):

- **Descartadas**: `device_type_encoded` (importancia < 1%), `tx_velocity_1h` (redundante con `tx_count_1h`, r=1.0)
- **Conservadas**: 5 directas + 5 ventana + 5 perfil histórico
