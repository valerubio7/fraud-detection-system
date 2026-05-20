# Seed de transacciones en TimescaleDB

Genera transacciones sintéticas e las inserta en la hypertable `public.transactions`.

## Variables de entorno requeridas

El script lee la configuración desde el `.env` del proyecto:

- `TIMESCALE_HOST` (default: `localhost`)
- `TIMESCALE_PORT` (default: `5432`)
- `TIMESCALE_USER` (default: `postgres`)
- `TIMESCALE_PASSWORD` (default: `postgres`)
- `TIMESCALE_DB` (default: `timescaledb`)

## Uso con Docker (recomendado)

Asegurate de tener el stack levantado (`docker compose up -d timescaledb`), luego cargá las variables del `.env` y ejecutá:

```bash
set -a; source .env; set +a

docker compose run --rm --no-deps --entrypoint python \
  -v "$(pwd)":/app -w /app \
  -e PYTHONPATH=/app \
  -e TIMESCALE_HOST=timescaledb \
  -e TIMESCALE_PORT=5432 \
  -e TIMESCALE_USER="${TIMESCALE_USER}" \
  -e TIMESCALE_PASSWORD="${TIMESCALE_PASSWORD}" \
  -e TIMESCALE_DB="${TIMESCALE_DB}" \
  airflow-webserver \
  database/timescaledb/seeds/seed_transactions.py \
  --count 10000 --fraud-rate 0.02 --seed 42 --batch-size 500
```

## Uso local

Necesitas tener Python y `psycopg2` instalados (puede ser via `uv`). Asegurate de que TimescaleDB este levantado y expuesto en `localhost:5433`.

```bash
set -a; source .env; set +a
export TIMESCALE_HOST=localhost
export TIMESCALE_PORT=5433
export PYTHONPATH="$(pwd)"

uv run python database/timescaledb/seeds/seed_transactions.py \
  --count 10000 --fraud-rate 0.02 --seed 42 --batch-size 500
```

## Parámetros

| Parámetro      | Default | Descripción                              |
|----------------|---------|------------------------------------------|
| `--count`      | 10000   | Cantidad de transacciones a insertar     |
| `--fraud-rate` | 0.02    | Proporción de fraude (0-1)               |
| `--seed`       | 42      | Seed para reproducibilidad               |
| `--batch-size` | 500     | Tamaño del batch de inserción            |

## Verificación (SQL)

```sql
-- Total de registros y fraude
SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE is_fraud IS TRUE) AS fraude
FROM public.transactions;

-- Distribución por país
SELECT country, COUNT(*)
FROM public.transactions
GROUP BY country
ORDER BY COUNT(*) DESC;

-- Rango de fechas
SELECT MIN("timestamp") AS min_ts, MAX("timestamp") AS max_ts
FROM public.transactions;

-- Ejemplo de verificación de ráfaga (frecuencia alta)
SELECT user_id, COUNT(*) AS tx_count
FROM public.transactions
WHERE "timestamp" >= NOW() - INTERVAL '30 minutes'
GROUP BY user_id
HAVING COUNT(*) >= 5
ORDER BY tx_count DESC;
```
