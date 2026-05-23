# mlops/

Contiene toda la automatización del ciclo de vida del modelo de detección de fraude. Se encarga de tres responsabilidades principales:

- **Entrenamiento y promoción:** un DAG reentrena el modelo diariamente con datos frescos de TimescaleDB y otro evalúa si el nuevo modelo supera al champion antes de promoverlo a producción.
- **Monitoreo de drift:** cada 6 horas compara la distribución de las features de producción contra el dataset de referencia del entrenamiento (data drift) y recalcula las métricas reales del modelo sobre predicciones etiquetadas (model drift). Si la degradación supera los umbrales configurados, dispara un reentrenamiento automático.
- **Calidad de datos:** un DAG horario verifica que el volumen de transacciones y predicciones esté dentro de rangos esperados y alerta si la distribución de montos es anómala.

Los tres módulos — `mlflow/`, `airflow/` y `evidently/` — trabajan juntos: MLflow almacena modelos y artefactos, Airflow orquesta la ejecución y Evidently provee el análisis estadístico de drift.

## Flujo general

```
retrain_fraud_model  (diario 02:00 UTC)
    └──► validate_and_promote_model  (event-driven)
              └──► promueve o archiva el modelo en MLflow

drift_detection_report  (cada 6 horas)
    ├── data drift + model drift en paralelo
    └── si severity HIGH/CRITICAL → dispara retrain_fraud_model

data_quality_check  (cada hora)
    └── alerta si hay pocas transacciones o distribución anómala
```

---

## mlflow/

| Archivo | Qué hace |
|---|---|
| `init_mlflow.py` | Bootstrap idempotente: crea el experimento `fraud-detection-v1` y registra el modelo `FraudDetectionModel` en el MLflow Registry. Se ejecuta una vez al levantar el stack. |

---

## airflow/dags/

| DAG | Schedule | Qué hace |
|---|---|---|
| `retrain_fraud_model` | `0 2 * * *` | Verifica que haya ≥ 1000 filas en TimescaleDB, lanza `model/train.py` y dispara `validate_and_promote_model`. |
| `validate_and_promote_model` | event-driven | Corre quality gates (F1 ≥ 0.85, AUC-ROC ≥ 0.90), compara challenger vs. champion y promueve o archiva en MLflow + PostgreSQL. |
| `drift_detection_report` | `0 */6 * * *` | Analiza data drift y model drift, persiste el reporte y dispara reentrenamiento si la severidad es HIGH o CRITICAL. |
| `data_quality_check` | `0 * * * *` | Verifica volumen de transacciones, tasa de predicciones y distribución de montos. Alerta si algo está fuera de rango. |

## airflow/plugins/

| Operador | Qué hace |
|---|---|
| `TimescaleExtractOperator` | Ejecuta SQL contra TimescaleDB y serializa el resultado a Parquet. Devuelve la ruta via XCom. |
| `MLflowRegisterModelOperator` | Transiciona una versión de modelo a un stage en el MLflow Registry. |
| `EvidentlyReportOperator` | Lee dos Parquets desde XCom, corre `DataDriftPreset` y devuelve drift score y resultados por feature via XCom. |

---

## evidently/

| Archivo | Qué hace |
|---|---|
| `reference_data.py` | Descarga `reference_dataset.parquet` desde los artefactos del run de MLflow activo. |
| `data_drift.py` | Corre `DataDriftPreset` de Evidently AI comparando features de producción vs. referencia. Devuelve `DataDriftResult` con drift por feature. |
| `model_drift.py` | Calcula F1/precision/recall sobre predicciones etiquetadas de los últimos 7 días y los compara contra el baseline de entrenamiento. Drift si `ΔF1 < -0.05`. |
| `drift_policy.py` | Combina los resultados y determina la severidad (`INFO / WARNING / HIGH / CRITICAL`). Si es HIGH o CRITICAL llama a la API de Airflow para disparar reentrenamiento. |
| `drift_store.py` | Persiste reportes en `drift_reports` y alertas en `alert_log` de PostgreSQL. |
| `report_uploader.py` | Sube el reporte HTML de Evidently como artefacto al run de MLflow correspondiente. |

### Severidad de drift

| Condición | Severidad | Reentrenamiento |
|---|---|---|
| Feature crítica con drift **y** model drift | `CRITICAL` | Sí |
| Feature crítica con drift **o** model drift | `HIGH` | Sí |
| Drift global > 30 % | `WARNING` | No |
| Sin umbral superado | `INFO` | No |

Features críticas: `tx_count_1h`, `amount_sum_1h`, `amount_ratio_vs_user_avg`, `is_country_new`, `seconds_since_last_tx`.
