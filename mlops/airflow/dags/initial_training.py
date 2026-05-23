import logging
import os
import re
import subprocess
import sys
from datetime import datetime

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

OUTPUT_DIR = "/tmp/airflow_model"
PROMOTE_DAG_ID = "validate_and_promote_model"


@dag(
    dag_id="initial_training",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
    default_args={"retries": 0},
    max_active_runs=1,
)
def initial_training() -> None:
    @task
    def run_training() -> dict:
        import mlflow
        from mlflow.tracking import MlflowClient

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

    train_result = run_training()
    trigger_validation(train_result)


initial_training()
