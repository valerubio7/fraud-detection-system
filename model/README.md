# model/

Módulo de entrenamiento, evaluación y promoción del modelo de detección de fraude. Implementa el ciclo de vida completo del modelo: desde el entrenamiento con datos históricos hasta su registro en MLflow y promoción a producción.

El pipeline tiene tres etapas secuenciales orquestadas por Airflow:

```
train → evaluate → promote
```

---

## Estructura

```
model/
  pipeline/       # Scripts ejecutables del pipeline
    train.py
    evaluate.py
    promote.py
  utils/          # Utilidades internas
    metrics.py
    plots.py
    tuning.py
    selected_features.py
```

---

## pipeline/

### `train.py`

Entrena un clasificador XGBoost sobre las transacciones históricas de TimescaleDB. El flujo es:

1. Carga transacciones etiquetadas desde TimescaleDB
2. Construye features con `offline_features.featurizer.TransactionFeaturizer`
3. Valida que las features seleccionadas coincidan con `utils/selected_features.py`
4. Divide los datos temporalmente (70% train / 15% val / 15% test)
5. Opcionalmente corre tuning de hiperparámetros con Optuna (`--tune`)
6. Entrena el modelo y busca el threshold óptimo por costo mínimo (FN/FP)
7. Evalúa en el set de test y genera artefactos (plots, métricas, encoder)
8. Registra todo en MLflow y deja el modelo en stage `Staging`

```bash
docker compose run --rm model python -m model.pipeline.train
docker compose run --rm model python -m model.pipeline.train --tune --n-trials 30
docker compose run --rm model python -m model.pipeline.train --limit 50000 --cost-fn 100 --cost-fp 5
```

**Artefactos generados** en `artifacts/model/`:

| Archivo | Contenido |
|---|---|
| `xgboost_model.joblib` | Modelo serializado |
| `categorical_encoder.joblib` | Encoder de features categóricas |
| `training_metadata.json` | Parámetros, métricas y fechas del run |
| `evaluation_results.json` | Métricas detalladas en el set de test |
| `reference_dataset.parquet` | Dataset de referencia para detección de drift |
| `confusion_matrix.png` | Matriz de confusión |
| `roc_curve.png` | Curva ROC |
| `pr_curve.png` | Curva Precision-Recall |
| `feature_importance.png` | Importancia de features (gain) |
| `threshold_analysis.png` | F1/Precision/Recall por threshold |

---

### `evaluate.py`

Corre los quality gates sobre un modelo en stage `Staging` y lo compara contra el modelo en `Production` (champion). Se ejecuta automáticamente después de `train.py` vía Airflow.

**Quality gates** (umbrales mínimos):

| Métrica | Umbral |
|---|---|
| F1-score (fraude) | ≥ 0.85 |
| AUC-ROC | ≥ 0.90 |
| Latencia P99 | ≤ 50 ms |

**Comparación champion/challenger**: el challenger debe superar al champion en al menos 0.02 de F1 para ganar.

```bash
docker compose run --rm model python -m model.pipeline.evaluate \
  --model-name FraudDetectionModel --model-version 2

# Con comparación contra el modelo en producción:
docker compose run --rm model python -m model.pipeline.evaluate \
  --model-name FraudDetectionModel --model-version 2 --compare
```

Sale con código 1 si los quality gates fallan, para que Airflow lo marque como tarea fallida.

---

### `promote.py`

Promueve un modelo desde `Staging` a `Production` en MLflow y registra el deployment en la tabla `model_deployments` de PostgreSQL. Llama a la stored procedure `activate_model_version()` que desactiva la versión anterior.

```bash
docker compose run --rm model python -m model.pipeline.promote \
  --model-name FraudDetectionModel --model-version 2
```

Solo acepta modelos que estén en stage `Staging`. Si ya existe un registro del mismo `run_id` en la base de datos, lo reutiliza (idempotente).

---

## utils/

### `metrics.py`

Funciones puras de evaluación usadas por `train.py` y `evaluate.py`:

- `evaluate_model()` — calcula precision, recall, F1, AUC-ROC, PR-AUC, matriz de confusión y métricas de costo dado un threshold
- `find_optimal_threshold()` — encuentra el threshold que minimiza el costo total `FN × cost_fn + FP × cost_fp`
- `compute_threshold_metrics()` — calcula F1/precision/recall para un rango de thresholds

### `plots.py`

Genera y guarda los gráficos del entrenamiento como PNG:

- `save_confusion_matrix_plot()`
- `save_roc_curve_plot()`
- `save_pr_curve_plot()`
- `save_feature_importance_plot()`
- `save_threshold_analysis_plot()`

Usa `matplotlib` en modo no interactivo (`Agg`).

### `tuning.py`

Ejecuta un estudio de Optuna para búsqueda de hiperparámetros de XGBoost. Se activa cuando `train.py` recibe `--tune`. Optimiza PR-AUC en el set de validación usando el sampler TPE. Registra cada trial como un run anidado en MLflow si el tracking está disponible.

### `selected_features.py`

Define `SELECTED_FEATURES`, la lista canónica de las 17 features que usa el modelo. `train.py` valida que la selección automática de features coincida con esta lista antes de continuar — si no coincide, falla con un error explícito.

Para regenerar la lista ante un cambio de dataset:
```bash
python offline_features/feature_selection.py
# Copiar report.selected_features en SELECTED_FEATURES
```

---

## Orquestación

En producción estos scripts no se corren manualmente — Airflow los orquesta mediante dos DAGs:

| DAG | Trigger | Qué hace |
|---|---|---|
| `retrain_fraud_model` | Diario 2am | Verifica datos disponibles y corre `train.py` |
| `validate_and_promote_model` | Disparado por retrain | Corre `evaluate.py` y `promote.py` si los gates pasan |

Los comandos manuales sirven para desarrollo, debug o el primer entrenamiento al iniciar el sistema desde cero.
