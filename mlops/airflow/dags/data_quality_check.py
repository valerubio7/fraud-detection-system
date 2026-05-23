import logging
import os
from datetime import datetime

import psycopg2
from airflow.decorators import dag, task

DAG_ID = "data_quality_check"
logger = logging.getLogger(__name__)


def _pg_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgresql"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "fraud_metadata_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB", "fraud_metadata"),
    )


def _ts_conn():
    return psycopg2.connect(
        host=os.getenv("TIMESCALE_HOST", "timescaledb"),
        port=int(os.getenv("TIMESCALE_PORT", "5432")),
        user=os.getenv("TIMESCALE_USER", "fraud_timeseries_user"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB", "fraud_transactions_timeseries"),
    )


def _insert_alert(conn, alert_type: str, severity: str, message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.alert_log (alert_type, severity, message) VALUES (%s, %s, %s)",
            (alert_type, severity, message),
        )
    conn.commit()


@dag(
    dag_id=DAG_ID,
    schedule="0 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "data-quality", "monitoring"],
    default_args={"retries": 0},
)
def data_quality_check() -> None:
    @task
    def check_transaction_volume() -> dict:
        threshold = int(os.getenv("MIN_TRANSACTIONS_PER_HOUR", "10"))

        conn_ts = _ts_conn()
        try:
            with conn_ts.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.transactions WHERE timestamp >= NOW() - INTERVAL '1 hour'")
                count = cur.fetchone()[0]
        finally:
            conn_ts.close()

        alert_triggered = count < threshold
        if alert_triggered:
            conn_pg = _pg_conn()
            try:
                _insert_alert(
                    conn_pg,
                    alert_type="LOW_TRANSACTION_VOLUME",
                    severity="WARNING",
                    message=f"Only {count} transactions in last hour (min: {threshold})",
                )
            finally:
                conn_pg.close()

        return {"transaction_count": int(count), "alert_triggered": alert_triggered}

    @task
    def check_prediction_rate() -> dict:
        threshold = int(os.getenv("MIN_PREDICTIONS_PER_HOUR", "5"))

        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.predictions_history WHERE timestamp >= NOW() - INTERVAL '1 hour'"
                )
                count = cur.fetchone()[0]

            alert_triggered = count < threshold
            if alert_triggered:
                _insert_alert(
                    conn,
                    alert_type="LOW_PREDICTION_RATE",
                    severity="WARNING",
                    message=f"Only {count} predictions in last hour (min: {threshold})",
                )
        finally:
            conn.close()

        return {"prediction_count": int(count), "alert_triggered": alert_triggered}

    @task
    def check_amount_distribution() -> dict:
        conn = _ts_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*), AVG(amount), STDDEV(amount), MIN(amount), MAX(amount) "
                    "FROM public.transactions "
                    "WHERE timestamp >= NOW() - INTERVAL '24 hours'"
                )
                row = cur.fetchone()
        finally:
            conn.close()

        row_count, avg_amount, std_amount, min_amount, max_amount = row

        if row_count is None or row_count < 10:
            return {"status": "insufficient_data"}

        avg_val = float(avg_amount)
        std_val = float(std_amount) if std_amount is not None else 0.0
        max_val = float(max_amount)

        alert_triggered = avg_val < 1.0 or max_val > 100_000.0
        if alert_triggered:
            conn_pg = _pg_conn()
            try:
                _insert_alert(
                    conn_pg,
                    alert_type="ANOMALOUS_AMOUNT_DISTRIBUTION",
                    severity="HIGH",
                    message=f"Anomalous amount distribution: avg={avg_val:.2f}, max={max_val:.2f}",
                )
            finally:
                conn_pg.close()

        return {"avg_amount": avg_val, "std_amount": std_val, "max_amount": max_val, "alert_triggered": alert_triggered}

    @task
    def summarize_checks(tx_result: dict, pred_result: dict, amount_result: dict) -> None:
        tx_count = tx_result.get("transaction_count", 0)
        pred_count = pred_result.get("prediction_count", 0)

        if amount_result.get("status") == "insufficient_data":
            avg_str = "N/A (insufficient data)"
        else:
            avg_str = f"{amount_result.get('avg_amount', 0.0):.2f}"

        logger.info(
            "Data quality check — transactions: %d/h, predictions: %d/h, amount_avg: %s", tx_count, pred_count, avg_str
        )

    tx = check_transaction_volume()
    pred = check_prediction_rate()
    amount = check_amount_distribution()
    summarize_checks(tx, pred, amount)


data_quality_check()
