import logging
import os
import re
import subprocess
import sys
from datetime import datetime

import psycopg2
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

logger = logging.getLogger(__name__)

MIN_ROW_COUNT = 1000
OUTPUT_DIR = "/tmp/airflow_model"
PROMOTE_DAG_ID = "validate_and_promote_model"


@dag(
    dag_id="retrain_fraud_model",
    schedule="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
    default_args={"retries": 0},
)
def retrain_fraud_model() -> None:
    @task
    def validate_data_availability() -> dict:
        conn = psycopg2.connect(
            host=os.getenv("TIMESCALE_HOST", "timescaledb"),
            port=int(os.getenv("TIMESCALE_PORT", "5432")),
            user=os.getenv("TIMESCALE_USER", "fraud_timeseries_user"),
            password=os.getenv("TIMESCALE_PASSWORD"),
            dbname=os.getenv("TIMESCALE_DB", "fraud_transactions_timeseries"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                    "FROM public.transactions WHERE is_fraud IS NOT NULL"
                )
                row_count, data_from, data_to = cur.fetchone()
        finally:
            conn.close()

        if row_count < MIN_ROW_COUNT:
            raise AirflowSkipException(
                f"Only {row_count} labeled transactions available "
                f"(minimum required: {MIN_ROW_COUNT}). Skipping retraining."
            )

        return {"row_count": int(row_count), "data_from": str(data_from), "data_to": str(data_to)}

    @task
    def run_training(data_info: dict) -> dict:
        import mlflow
        from mlflow.tracking import MlflowClient

        logger.info(
            "Training with %d rows (%s → %s)", data_info["row_count"], data_info["data_from"], data_info["data_to"]
        )

        result = subprocess.run(
            [
                sys.executable,
                "/opt/airflow/project/model/pipeline/train.py",
                "--seed",
                "42",
                "--output-dir",
                OUTPUT_DIR,
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "/opt/airflow/project"},
        )

        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)

        run_id = None
        for line in result.stdout.splitlines():
            match = re.search(r"MLflow run_id:\s*(\S+)", line)
            if match:
                run_id = match.group(1)
                break
        if run_id is None:
            logger.warning("Could not parse run_id from train.py output.")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()
        model_name = os.getenv("MODEL_NAME", "FraudDetectionModel")
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not versions:
            raise RuntimeError(f"No Staging version found for '{model_name}' after training.")
        model_version = versions[0].version

        return {"run_id": run_id or "", "model_version": model_version, "model_name": model_name}

    @task
    def trigger_validation(train_result: dict) -> None:
        from airflow.api.common.trigger_dag import trigger_dag

        model_version = train_result["model_version"]
        if not model_version:
            raise RuntimeError("model_version is empty — cannot trigger validation.")

        trigger_dag(
            dag_id=PROMOTE_DAG_ID,
            conf={
                "model_name": train_result["model_name"],
                "model_version": model_version,
                "run_id": train_result["run_id"],
            },
            replace_microseconds=False,
        )

    data_info = validate_data_availability()
    train_result = run_training(data_info)
    trigger_validation(train_result)


retrain_fraud_model()
