# Database

Esquemas, migraciones, seeds, funciones y triggers para PostgreSQL (capa operacional/MLflow) y TimescaleDB (transacciones time-series).

## Estructura

```
database/
├── postgresql/                   # Datos operacionales y ML metadata
│   ├── migrations/
│   │   └── 001_initial_schema.sql
│   ├── stored_procedures/
│   │   └── 001_initial_stored_procedures.sql
│   └── triggers/
│       └── 001_initial_triggers.sql
└── timescaledb/                  # Transacciones time-series
    ├── migrations/
    │   └── 001_initial_schema.sql
    └── seeds/
        └── seed_transactions.py
```

Las migraciones se ejecutan via `scripts/setup.sh` en orden alfabético. Son idempotentes (`IF NOT EXISTS`, `DROP TRIGGER IF EXISTS` antes de `CREATE`).

## PostgreSQL — Tablas operacionales

| Tabla | Propósito |
|---|---|
| `model_deployments` | Registry de versiones de modelos desplegados: nombre, versión, MLflow run ID, métricas (F1, precision, recall, AUC-ROC), ventana de training, flag `is_active` |
| `predictions_history` | Historial de inferencias: `transaction_id`, `model_version_id`, `prediction_score`, `prediction_label`, `actual_label` (opcional), `latency_ms` |
| `drift_reports` | Resultados de análisis de drift: `drift_score`, `feature_drifts` (JSONB), `alert_triggered`, `remediation_action` |
| `alert_log` | Log de alertas operacionales: tipo, severidad (`INFO`/`WARNING`/`HIGH`/`CRITICAL`), mensaje, acknowledgment |
| `audit_log` | Traza de auditoría genérica vía triggers: tabla, operación, usuario, old/new values (JSONB) |

### Stored Procedures

| Función | Descripción |
|---|---|
| `activate_model_version(p_model_version_id)` | Desactiva todos los otros modelos activos, activa el especificado, logea en `audit_log` (usa `FOR UPDATE` para concurrencia) |
| `calculate_model_metrics(p_model_version_id, p_date_from, p_date_to)` | Computa precision, recall, F1 comparando `prediction_label` vs `actual_label` (mínimo 100 registros) |

### Triggers

| Trigger | Evento | Función |
|---|---|---|
| `alert_on_high_fraud_rate` | `AFTER INSERT` en `predictions_history` | Si la tasa de fraude de los últimos 15 min supera 5%, inserta alerta y emite `pg_notify('fraud_alerts', payload)` |
| `audit_model_deployments` | `AFTER INSERT OR UPDATE OR DELETE` en `model_deployments` | Serializa old/new state en `audit_log` |
| `audit_predictions_history` | `AFTER INSERT OR UPDATE OR DELETE` en `predictions_history` | Idem |

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

`seed_transactions.py` genera transacciones sintéticas para desarrollo:

```bash
docker compose run --rm seed python database/timescaledb/seeds/seed_transactions.py --count 50000 --fraud-rate 0.02
```

Usa los mismos generadores que `ingestion.producer.generator` y los 4 patrones de fraude (amount_anomaly, unusual_country, high_frequency, unknown_merchant). Idempotente vía `ON CONFLICT DO NOTHING`.
