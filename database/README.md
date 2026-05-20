# Database

Esquemas, migraciones, seeds, funciones y triggers para PostgreSQL (capa operacional/MLflow) y TimescaleDB (transacciones time-series).

Las migraciones se ejecutan via `scripts/setup.sh` en orden alfabético. Son idempotentes (`IF NOT EXISTS`, `DROP TRIGGER IF EXISTS` antes de `CREATE`).

## PostgreSQL — Tablas operacionales

| Tabla | Propósito |
|---|---|
| `model_deployments` | Registry de versiones de modelos desplegados: nombre, versión, MLflow run ID, métricas (F1, precision, recall, AUC-ROC), ventana de training, flag `is_active` |
| `predictions_history` | Historial de inferencias: `transaction_id`, `model_version_id`, `prediction_score`, `prediction_label`, `actual_label` (opcional), `latency_ms` |
| `drift_reports` | Resultados de análisis de drift: `drift_score`, `feature_drifts` (JSONB), `alert_triggered`, `remediation_action` |
| `alert_log` | Log de alertas operacionales: tipo, severidad (`INFO`/`WARNING`/`HIGH`/`CRITICAL`), mensaje, acknowledgment |

### Indexes

| Tabla | Índice | Tipo | Columnas |
|---|---|---|---|
| `model_deployments` | `idx_model_deployments_active` | B-tree parcial `WHERE is_active IS TRUE` | `created_at DESC` |
| `predictions_history` | `idx_predictions_history_model_version_id` | B-tree | `model_version_id` |
| `predictions_history` | `idx_predictions_history_timestamp` | B-tree | `timestamp` |
| `predictions_history` | `idx_predictions_history_transaction_id` | B-tree | `transaction_id` |
| `drift_reports` | `idx_drift_reports_report_date` | B-tree | `report_date` |
| `drift_reports` | `idx_drift_reports_model_version_id` | B-tree | `model_version_id` |
| `drift_reports` | `idx_drift_reports_alerts_true` | B-tree parcial `WHERE alert_triggered IS TRUE` | `report_date DESC` |
| `alert_log` | `idx_alert_log_triggered_at` | B-tree | `triggered_at DESC` |
| `alert_log` | `idx_alert_log_open_unack` | B-tree parcial `WHERE acknowledged_at IS NULL` | `triggered_at DESC` |
| `alert_log` | `idx_alert_log_severity` | B-tree | `severity` |

### Stored Procedures

| Función | Descripción |
|---|---|
| `activate_model_version(p_model_version_id)` | Desactiva todos los otros modelos activos y activa el especificado (usa `FOR UPDATE` para concurrencia) |

### Triggers

| Trigger | Evento | Función |
|---|---|---|
| `alert_on_high_fraud_rate` | `AFTER INSERT` en `predictions_history` | Si la tasa de fraude de los últimos 15 min supera 5%, inserta alerta y emite `pg_notify('fraud_alerts', payload)` |

## TimescaleDB — Transacciones time-series

### `public.transactions` (hypertable)

Almacena todas las transacciones crudas con particionado diario por `timestamp`.

| Columna | Tipo |
|---|---|
| `transaction_id` | `UUID` (PK compuesta con timestamp) |
| `user_id` | `TEXT` |
| `merchant_id` | `TEXT` |
| `merchant_category` | `TEXT` |
| `amount` | `NUMERIC(12,2)` |
| `country`, `device_type`, `ip_hash` | `TEXT` |
| `timestamp` | `TIMESTAMPTZ` (PK) |
| `is_fraud` | `BOOLEAN` (nullable) |
| `model_score` | `DOUBLE PRECISION` (nullable) |
| `latency_ms` | `DOUBLE PRECISION` (nullable) |

### Indexes

| Índice | Tipo | Columnas |
|---|---|---|
| `transactions_pkey` | Primary Key compuesta (requerida por hypertable) | `(transaction_id, timestamp)` |
| `transactions_user_timestamp_idx` | B-tree compuesto | `(user_id, timestamp)` |
| `transactions_timestamp_idx` | B-tree | `(timestamp)` |
| `transactions_is_fraud_true_idx` | B-tree parcial `WHERE is_fraud IS TRUE` | `(timestamp)` |

### Continuous Aggregates

| Vista | Bucket | Métricas | Refresco |
|---|---|---|---|
| `fraud_volume_hourly` | 1 hora | `COUNT(*)`, `COUNT(*) FILTER (WHERE is_fraud)`, `fraud_rate` | Cada 5 min, últimos 30 días |
| `merchant_amount_daily` | 1 día | `SUM(amount)`, `COUNT(*)` por merchant | Cada 5 min, últimos 30 días |

### Policies

| Policy | Detalle |
|---|---|
| **Compression** | Chunks > 7 días: comprimidos, segmentados por `user_id`, ordenados por `timestamp DESC` |
| **Retention** | Datos > 2 años: drop automático |

### Seeds

`seed_transactions.py` genera transacciones sintéticas para desarrollo.
