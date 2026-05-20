-- ============================================================================
-- Base table and hypertable
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.transactions (
    transaction_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    merchant_category TEXT,
    amount NUMERIC(12, 2) NOT NULL,
    country TEXT,
    device_type TEXT,
    ip_hash TEXT,
    "timestamp" TIMESTAMPTZ NOT NULL,
    is_fraud BOOLEAN,
    model_score DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION,
    -- TimescaleDB hypertables require unique indexes/PKs to include the time column.
    PRIMARY KEY (transaction_id, "timestamp")
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'public'
          AND hypertable_name = 'transactions'
    ) THEN
        PERFORM create_hypertable(
            'public.transactions',
            'timestamp',
            chunk_time_interval => INTERVAL '1 day'
        );
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END
$$;


CREATE INDEX IF NOT EXISTS transactions_user_timestamp_idx
    ON public.transactions (user_id, "timestamp");

CREATE INDEX IF NOT EXISTS transactions_timestamp_idx
    ON public.transactions ("timestamp");

CREATE INDEX IF NOT EXISTS transactions_is_fraud_true_idx
    ON public.transactions ("timestamp")
    WHERE is_fraud IS TRUE;

-- ============================================================================
-- Continuous aggregates and refresh policies
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS public.fraud_volume_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 hour', "timestamp") AS bucket_hour,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE is_fraud IS TRUE) AS total_fraud_transactions,
    (COUNT(*) FILTER (WHERE is_fraud IS TRUE))::DOUBLE PRECISION / NULLIF(COUNT(*), 0) AS fraud_rate
FROM public.transactions
GROUP BY 1
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS public.merchant_amount_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 day', "timestamp") AS bucket_day,
    merchant_id,
    SUM(amount) AS total_amount,
    COUNT(*) AS transaction_count
FROM public.transactions
GROUP BY 1, 2
WITH NO DATA;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM timescaledb_information.jobs AS job
        JOIN timescaledb_information.continuous_aggregates AS cagg
          ON job.hypertable_schema = cagg.materialization_hypertable_schema
         AND job.hypertable_name = cagg.materialization_hypertable_name
        WHERE job.proc_name = 'policy_refresh_continuous_aggregate'
          AND cagg.view_schema = 'public'
          AND cagg.view_name = 'fraud_volume_hourly'
    ) THEN
        PERFORM add_continuous_aggregate_policy(
            'public.fraud_volume_hourly',
            start_offset => INTERVAL '30 days',
            end_offset => INTERVAL '5 minutes',
            schedule_interval => INTERVAL '5 minutes'
        );
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM timescaledb_information.jobs AS job
        JOIN timescaledb_information.continuous_aggregates AS cagg
          ON job.hypertable_schema = cagg.materialization_hypertable_schema
         AND job.hypertable_name = cagg.materialization_hypertable_name
        WHERE job.proc_name = 'policy_refresh_continuous_aggregate'
          AND cagg.view_schema = 'public'
          AND cagg.view_name = 'merchant_amount_daily'
    ) THEN
        PERFORM add_continuous_aggregate_policy(
            'public.merchant_amount_daily',
            start_offset => INTERVAL '30 days',
            end_offset => INTERVAL '5 minutes',
            schedule_interval => INTERVAL '5 minutes'
        );
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END
$$;

-- ============================================================================
-- Compression
-- ============================================================================

ALTER TABLE IF EXISTS public.transactions
SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id',
    timescaledb.compress_orderby = '"timestamp" DESC'
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM timescaledb_information.jobs
        WHERE hypertable_schema = 'public'
          AND hypertable_name = 'transactions'
          AND proc_name = 'policy_compression'
    ) THEN
        PERFORM add_compression_policy(
            'public.transactions',
            compress_after => INTERVAL '7 days'
        );
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END
$$;

-- ============================================================================
-- Retention
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM timescaledb_information.jobs
        WHERE hypertable_schema = 'public'
          AND hypertable_name = 'transactions'
          AND proc_name = 'policy_retention'
    ) THEN
        PERFORM add_retention_policy(
            'public.transactions',
            drop_after => INTERVAL '2 years'
        );
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END
$$;
