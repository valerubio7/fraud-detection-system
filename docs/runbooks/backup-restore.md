# Runbook: Backup y restauración de datos

## Datos persistentes del sistema

| Volumen Docker | Contenido | Criticidad |
|---|---|---|
| `postgres-data` | model_deployments, predictions_history, drift_reports, alert_log, airflow_metadata | Alta |
| `timescale-data` | transactions (hypertable), fraud_volume_hourly (cagg) | Alta |
| `mlflow-data` | artefactos de modelos (joblib, encoders, plots) | Media |
| `grafana-data` | estado interno de Grafana (dashboards no provisionados) | Baja (provisionados automáticamente) |
| `prometheus-data` | métricas históricas (series temporales de 15 días) | Baja (se regeneran) |
| `airflow-logs` | logs de ejecución de DAGs | Baja |

## Backup de PostgreSQL (metadata + airflow)

```bash
source .env
# Backup completo de la base de datos de metadata:
docker compose exec -T postgresql pg_dump \
  -U "${POSTGRES_USER}" "${POSTGRES_DB}" \
  | gzip > "backup_postgresql_$(date +%Y%m%d_%H%M%S).sql.gz"

# Backup de airflow_metadata:
docker compose exec -T postgresql pg_dump \
  -U "${POSTGRES_USER}" airflow_metadata \
  | gzip > "backup_airflow_$(date +%Y%m%d_%H%M%S).sql.gz"
```

## Backup de TimescaleDB

```bash
source .env
# TimescaleDB acepta pg_dump estándar.
# El flag --no-tablespaces evita errores con tablespaces custom de TimescaleDB:
docker compose exec -T timescaledb pg_dump \
  -U "${TIMESCALE_USER}" "${TIMESCALE_DB}" \
  --no-tablespaces \
  | gzip > "backup_timescaledb_$(date +%Y%m%d_%H%M%S).sql.gz"
```

## Backup de artefactos de MLflow

```bash
# Los artefactos están en el volumen mlflow-data montado en /mlflow/artifacts:
docker run --rm \
  -v fraud-detection-system_mlflow-data:/data \
  -v "$(pwd)":/backup \
  alpine tar czf /backup/backup_mlflow_$(date +%Y%m%d_%H%M%S).tar.gz /data
```

## Restauración de PostgreSQL

⚠️ La restauración sobreescribe los datos existentes.

```bash
source .env

# 1. Detener servicios que usan la base de datos:
docker compose stop fastapi airflow-webserver airflow-scheduler mlflow

# 2. Restaurar:
gunzip -c backup_postgresql_YYYYMMDD_HHMMSS.sql.gz \
  | docker compose exec -T postgresql psql \
      -U "${POSTGRES_USER}" "${POSTGRES_DB}"

# 3. Reiniciar servicios:
docker compose up -d fastapi airflow-webserver airflow-scheduler mlflow

# 4. Verificar:
curl -fsS http://localhost:8000/health
```

## Restauración de TimescaleDB

```bash
source .env

# 1. Detener consumer (escribe a TimescaleDB):
docker compose stop consumer producer

# 2. Restaurar (TimescaleDB requiere la extensión habilitada previamente):
docker compose exec -T timescaledb psql \
  -U "${TIMESCALE_USER}" "${TIMESCALE_DB}" \
  -c "SELECT timescaledb_pre_restore();"

gunzip -c backup_timescaledb_YYYYMMDD_HHMMSS.sql.gz \
  | docker compose exec -T timescaledb psql \
      -U "${TIMESCALE_USER}" "${TIMESCALE_DB}"

docker compose exec -T timescaledb psql \
  -U "${TIMESCALE_USER}" "${TIMESCALE_DB}" \
  -c "SELECT timescaledb_post_restore();"

# 3. Refrescar continuous aggregates:
docker compose exec -T timescaledb psql \
  -U "${TIMESCALE_USER}" "${TIMESCALE_DB}" \
  -c "CALL refresh_continuous_aggregate('fraud_volume_hourly', NULL, NULL);
      CALL refresh_continuous_aggregate('merchant_amount_daily', NULL, NULL);"

# 4. Reiniciar consumer:
docker compose up -d consumer
```

## Frecuencia recomendada de backup

| Base de datos | Frecuencia | Retención |
|---|---|---|
| PostgreSQL (metadata) | Diaria (2 AM) | 30 días |
| TimescaleDB | Diaria (3 AM) | 7 días (datos más antiguos tienen menos valor) |
| MLflow artifacts | Después de cada promoción de modelo | Indefinida |
