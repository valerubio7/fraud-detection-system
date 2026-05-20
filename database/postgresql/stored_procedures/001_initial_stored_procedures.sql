-- ============================================================================
-- activate_model_version: Deactivates all active models (except the specified
-- one) and activates the new one. Uses FOR UPDATE to prevent race conditions
-- on concurrent activations.
-- ============================================================================
CREATE OR REPLACE FUNCTION public.activate_model_version(p_model_version_id INTEGER)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_target RECORD;
    v_old_active RECORD;
BEGIN
    RAISE NOTICE 'activate_model_version start: model_version_id=%', p_model_version_id;

    SELECT mv.id, mv.model_name, mv.version, mv.is_active
    INTO v_target
    FROM public.model_deployments AS mv
    WHERE mv.id = p_model_version_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'activate_model_version failed: model_version_id % does not exist in model_deployments',
            p_model_version_id;
    END IF;

    SELECT mv.id, mv.model_name, mv.version, mv.is_active
    INTO v_old_active
    FROM public.model_deployments AS mv
    WHERE mv.is_active IS TRUE
      AND mv.id <> p_model_version_id
    ORDER BY mv.created_at DESC, mv.id DESC
    LIMIT 1
    FOR UPDATE;

    IF FOUND THEN
        RAISE NOTICE
            'Previous active model found: id=%, model_name=%, version=%',
            v_old_active.id, v_old_active.model_name, v_old_active.version;
    ELSE
        RAISE NOTICE 'No previously active model found (first activation case).';
    END IF;

    RAISE NOTICE 'Deactivating all active versions except id=%', p_model_version_id;
    UPDATE public.model_deployments
    SET is_active = FALSE
    WHERE is_active IS TRUE
      AND id <> p_model_version_id;

    RAISE NOTICE 'Activating model_version_id=%', p_model_version_id;
    UPDATE public.model_deployments
    SET is_active = TRUE
    WHERE id = p_model_version_id;

    RAISE NOTICE 'activate_model_version complete: model_version_id=%', p_model_version_id;
END;
$$;


-- ============================================================================
-- check_fraud_rate: AFTER INSERT trigger on predictions_history. If the fraud
-- rate in the last 15 minutes exceeds 5%, inserts an alert into alert_log
-- and emits pg_notify('fraud_alerts', payload). Skips if an unacknowledged
-- alert of the same type already exists within the window.
-- ============================================================================
CREATE OR REPLACE FUNCTION public.check_fraud_rate()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_fraud_threshold DOUBLE PRECISION := 0.05;
    v_window_minutes INTEGER := 15;
    v_window_interval INTERVAL := INTERVAL '15 minutes';

    v_reference_ts TIMESTAMPTZ := COALESCE(NEW."timestamp", NOW());
    v_total_count BIGINT := 0;
    v_fraud_count BIGINT := 0;
    v_fraud_rate DOUBLE PRECISION := 0.0;
    v_has_recent_open_alert BOOLEAN := FALSE;

    v_alert_message TEXT;
    v_notify_payload TEXT;
BEGIN
    SELECT
        COUNT(*)::BIGINT,
        COALESCE(SUM(CASE WHEN ph.prediction_label IS TRUE THEN 1 ELSE 0 END), 0)::BIGINT
    INTO
        v_total_count,
        v_fraud_count
    FROM public.predictions_history AS ph
    WHERE ph."timestamp" >= (v_reference_ts - v_window_interval)
      AND ph."timestamp" <= v_reference_ts;

    IF v_total_count > 0 THEN
        v_fraud_rate := v_fraud_count::DOUBLE PRECISION / v_total_count::DOUBLE PRECISION;
    ELSE
        v_fraud_rate := 0.0;
    END IF;

    RAISE NOTICE
        'check_fraud_rate: fraud_rate=%, fraud_count=%, total_count=%, window_minutes=%',
        v_fraud_rate,
        v_fraud_count,
        v_total_count,
        v_window_minutes;

    SELECT EXISTS (
        SELECT 1
        FROM public.alert_log AS al
        WHERE al.alert_type = 'HIGH_FRAUD_RATE'
          AND al.acknowledged_at IS NULL
          AND al.triggered_at >= (v_reference_ts - v_window_interval)
    )
    INTO v_has_recent_open_alert;

    IF v_fraud_rate > v_fraud_threshold AND NOT v_has_recent_open_alert THEN
        v_alert_message := format(
            'High fraud rate detected: %s%% over the last %s minutes (%s/%s predictions).',
            to_char(v_fraud_rate * 100.0, 'FM999990.00'),
            v_window_minutes,
            v_fraud_count,
            v_total_count
        );

        INSERT INTO public.alert_log (
            alert_type,
            severity,
            message,
            triggered_at
        )
        VALUES (
            'HIGH_FRAUD_RATE',
            'HIGH',
            v_alert_message,
            v_reference_ts
        );

        v_notify_payload := jsonb_build_object(
            'fraud_rate', round(v_fraud_rate::NUMERIC, 6),
            'window_minutes', v_window_minutes,
            'triggered_at', v_reference_ts
        )::TEXT;

        PERFORM pg_notify('fraud_alerts', v_notify_payload);

        RAISE NOTICE 'check_fraud_rate: HIGH_FRAUD_RATE alert inserted and pg_notify emitted.';
    END IF;

    RETURN NEW;
END;
$$;
