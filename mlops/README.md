# mlops/

Este directorio contiene la capa de automatización MLOps del sistema: inicialización de MLflow, DAGs de Airflow y módulos de detección de drift con Evidently AI.

```
mlops/
├── mlflow/
│   └── init_mlflow.py          # Bootstrap idempotente del servidor MLflow
├── airflow/
│   ├── dags/
│   │   ├── retrain_fraud_model.py          # Reentrenamiento diario
│   │   ├── validate_and_promote_model.py   # Validación y promoción (event-driven)
│   │   ├── drift_detection_report.py       # Detección de drift cada 6 horas
│   │   └── data_quality_check.py           # Calidad de datos cada hora
│   └── plugins/
│       └── operators.py                    # Operadores Airflow reutilizables
└── evidently/
    ├── data_drift.py       # Análisis de data drift con Evidently DataDriftPreset
    ├── model_drift.py      # Degradación de métricas de producción vs. baseline
    ├── drift_store.py      # Persistencia de reportes en PostgreSQL
    ├── drift_policy.py     # Umbrales de alerta y lógica de severidad
    ├── reference_data.py   # Carga del dataset de referencia desde MLflow
    └── report_uploader.py  # Subida de reportes HTML a MLflow como artefactos
```

## Ciclo de vida automatizado

Los cuatro DAGs forman un pipeline autónomo donde cada pieza dispara la siguiente:

```
TimescaleDB
    │
    ▼
retrain_fraud_model  (diario 02:00 UTC)
    │ dispara via trigger_dag
    ▼
validate_and_promote_model  (event-driven)
    ├── quality gates (F1 ≥ 0.85, AUC-ROC ≥ 0.90, P99 ≤ 50ms)
    ├── comparación challenger vs. champion (delta F1 > 0.02)
    └── promoción a Production + registro en PostgreSQL
          │
          └── si falla/pierde → archiva versión en MLflow

drift_detection_report  (cada 6 horas)
    ├── data drift   → Evidently DataDriftPreset sobre 16 features
    ├── model drift  → F1 producción vs. baseline de entrenamiento
    ├── persiste en drift_reports + alert_log
    ├── exporta reporte HTML a MLflow artifacts
    └── si severity HIGH/CRITICAL → dispara retrain_fraud_model

data_quality_check  (cada hora)
    ├── volumen de transacciones (< MIN_TRANSACTIONS_PER_HOUR → alerta)
    ├── tasa de predicciones   (< MIN_PREDICTIONS_PER_HOUR → alerta)
    └── distribución de montos (avg < 1 o max > 100 000 → alerta)
```

## DAGs

### `retrain_fraud_model`

**Schedule:** `0 2 * * *` (diario a las 02:00 UTC)

Valida que haya al menos 1000 transacciones etiquetadas en TimescaleDB, ejecuta `model/train.py` como subproceso e invoca `validate_and_promote_model` con el `model_version` recién creado. Si los datos son insuficientes, hace skip sin marcar el run como fallido.

### `validate_and_promote_model`

**Schedule:** `None` — event-driven, disparado por `retrain_fraud_model` o manualmente.

Recibe `model_name` y `model_version` vía `dag_run.conf`. Ejecuta en secuencia:

1. Quality gates sobre el test set temporal (último 20% cronológico de TimescaleDB).
2. Comparación F1 challenger vs. champion actual en `Production`.
3. Promoción atómica: primero transacciona en PostgreSQL, luego transiciona en MLflow.
4. Si algún paso falla o hace skip, archiva la versión (`TriggerRule.ONE_FAILED`).

### `drift_detection_report`

**Schedule:** `0 */6 * * *` (cada 6 horas)

Requiere un deployment activo en `model_deployments`. Corre data drift y model drift en paralelo, persiste el reporte combinado y dispara reentrenamiento si la severidad es `HIGH` o `CRITICAL`.

Configuración de umbrales vía variables de entorno:

| Variable | Default | Descripción |
|---|---|---|
| `DRIFT_THRESHOLD_CRITICAL` | `0.20` | Drift score de features críticas que activa severidad HIGH/CRITICAL |
| `DRIFT_THRESHOLD_GLOBAL` | `0.30` | Drift share global que activa severidad WARNING |
| `MODEL_F1_DEGRADATION_THRESHOLD` | `0.05` | Caída de F1 en producción que activa model drift |

**Features críticas:** `tx_count_1h`, `amount_sum_1h`, `amount_ratio_vs_user_avg`, `is_country_new`, `seconds_since_last_tx`.

### `data_quality_check`

**Schedule:** `0 * * * *` (cada hora)

Tres checks en paralelo. Los umbrales son configurables por variable de entorno:

| Variable | Default |
|---|---|
| `MIN_TRANSACTIONS_PER_HOUR` | `10` |
| `MIN_PREDICTIONS_PER_HOUR` | `5` |

## Operadores Airflow (`operators.py`)

| Operador | Descripción |
|---|---|
| `TimescaleExtractOperator` | Ejecuta SQL contra TimescaleDB o PostgreSQL y serializa el resultado a Parquet. Devuelve la ruta como XCom. |
| `MLflowRegisterModelOperator` | Transiciona una versión de modelo a un stage en el MLflow Registry. Lee el `model_version` desde XCom. |
| `EvidentlyReportOperator` | Lee dos Parquets desde XCom y ejecuta `DataDriftPreset`. Devuelve drift score y resultados por feature como XCom. |

## Módulos Evidently (`mlops/evidently/`)

### `data_drift.py`

Encapsula `DataDriftPreset`. Expone:

- `run_data_drift_report(ref, cur, columns)` → `DataDriftResult`
- `run_data_drift_report_with_html(ref, cur, columns)` → `(DataDriftResult, html_path)`

`DataDriftResult` incluye `drift_share` (proporción de features con drift), `dataset_drift` (booleano global) y un dict de `FeatureDriftResult` por feature con `drift_detected`, `drift_score` y el nombre del test estadístico.

### `model_drift.py`

Compara métricas de producción contra el baseline de entrenamiento:

- `fetch_labeled_predictions(deployment_id)` extrae predicciones con `actual_label` de los últimos 7 días.
- `run_model_drift_report(ref_metrics, labeled_df)` calcula F1/precisión/recall actuales y los compara. Drift detectado si `delta_F1 < -0.05`. Requiere mínimo 50 predicciones etiquetadas.

### `drift_policy.py`

`evaluate_drift_action(data_drift_result, model_drift_result)` → `DriftAction`:

| Condición | Severidad | Reentrenamiento |
|---|---|---|
| Features críticas con drift + model drift | `CRITICAL` | Sí |
| Features críticas con drift O model drift | `HIGH` | Sí |
| Drift global > `DRIFT_THRESHOLD_GLOBAL` | `WARNING` | No |
| Sin umbral superado | `INFO` | No |

### `drift_store.py`

`DriftReportStore.save()` persiste en `public.drift_reports` con los feature drifts serializados como JSONB (incluye tanto data drift como model drift en un objeto combinado). `save_alert()` inserta en `public.alert_log`.

### `reference_data.py`

`load_reference_dataset(run_id, tracking_uri)` descarga `reference_dataset.parquet` del artefacto MLflow del run de entrenamiento. Garantiza que el drift siempre se calcule contra los datos exactos con los que se entrenó la versión activa.

### `report_uploader.py`

`upload_report_to_mlflow(run_id, html_path, artifact_subfolder, tracking_uri)` sube el HTML al artifact store. Si falla, loggea un warning y retorna `None` sin interrumpir el DAG (degradación graceful).

## Inicialización (`mlops/mlflow/init_mlflow.py`)

El script es invocado por `scripts/setup.sh` tras el healthcheck de MLflow:

```bash
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app mlflow python mlops/mlflow/init_mlflow.py
```

Crea el experimento `fraud-detection-v1` con tags de proyecto y configura los metadatos del modelo `FraudDetectionModel` en el Registry. Es idempotente: puede correrse múltiples veces sin efectos destructivos.
