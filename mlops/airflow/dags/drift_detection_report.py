"""DAG de detección de drift cada 6 horas usando Evidently AI."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models.xcom_arg import XComArg

sys.path.insert(0, "/opt/airflow/project")

from fraud_operators import EvidentlyReportOperator, TimescaleExtractOperator

from mlops.evidently.reference import load_reference_dataset
from model.selected_features import SELECTED_FEATURES

DAG_ID = "drift_detection_report"
DRIFT_ALERT_THRESHOLD = 0.3
ENCODER_ARTIFACT_PATH = "categorical_encoder.joblib"
ENCODER_DST_DIR = "/tmp/drift_encoder"
REFERENCE_PARQUET = "/tmp/drift_reference.parquet"
PRODUCTION_PARQUET = "/tmp/drift_production.parquet"
MIN_PRODUCTION_ROWS = 100

_PRODUCTION_SQL = """
    SELECT transaction_id, user_id, merchant_id, merchant_category,
           amount, country, device_type, ip_hash, timestamp, is_fraud
    FROM public.transactions
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    ORDER BY timestamp
"""


def _pg_conn():
    import config

    s = config.postgres_settings
    return psycopg2.connect(host=s.host, port=s.port, user=s.user, password=s.password, dbname=s.db)


def _download_encoder(run_id: str) -> Path:
    import mlflow
    from mlflow.tracking import MlflowClient

    import config

    mlflow.set_tracking_uri(config.mlflow_settings.tracking_uri)
    client = MlflowClient()
    encoder_dir = Path(ENCODER_DST_DIR)
    encoder_dir.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(run_id, ENCODER_ARTIFACT_PATH, str(encoder_dir))
    return encoder_dir


@dag(
    dag_id=DAG_ID,
    schedule="0 */6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "drift", "monitoring"],
    default_args={"retries": 0},
)
def drift_detection_report() -> None:
    @task
    def fetch_active_deployment() -> dict:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, model_name, version, mlflow_run_id "
                    "FROM public.model_deployments WHERE is_active = TRUE LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise AirflowSkipException("No active model deployment found")

        deployment_id, model_name, model_version, mlflow_run_id = row
        return {
            "deployment_id": int(deployment_id),
            "model_name": str(model_name),
            "model_version": str(model_version),
            "mlflow_run_id": str(mlflow_run_id),
        }

    extract_production = TimescaleExtractOperator(
        task_id="fetch_production_data",
        sql=_PRODUCTION_SQL,
        output_path="/tmp/drift_raw_production.parquet",
    )

    @task
    def featurize_reference(active: dict) -> str:
        import config

        ref_df = load_reference_dataset(
            run_id=active["mlflow_run_id"],
            tracking_uri=config.mlflow_settings.tracking_uri,
        )
        ref_df[SELECTED_FEATURES].to_parquet(REFERENCE_PARQUET, index=False)
        return REFERENCE_PARQUET

    @task
    def featurize_production(active: dict, raw_path: str) -> str:
        import pandas as pd

        from feature_engineering.offline.featurizer import TransactionFeaturizer

        encoder_dir = _download_encoder(active["mlflow_run_id"])
        featurizer = TransactionFeaturizer(encoders_dir=encoder_dir)
        df = pd.read_parquet(raw_path).reset_index(drop=True)

        if len(df) < MIN_PRODUCTION_ROWS:
            raise AirflowSkipException(
                f"Insufficient production data for drift analysis (<{MIN_PRODUCTION_ROWS} rows)"
            )

        X = featurizer.transform(df)
        X[SELECTED_FEATURES].to_parquet(PRODUCTION_PARQUET, index=False)
        return PRODUCTION_PARQUET

    # Drift analysis — reads featurized Parquets via XCom from featurize tasks
    run_drift = EvidentlyReportOperator(
        task_id="run_evidently_report",
        reference_path_xcom_task_id="featurize_reference",
        current_path_xcom_task_id="featurize_production",
        columns=SELECTED_FEATURES,
    )

    @task
    def save_report_to_postgresql(deployment: dict, report: dict) -> None:
        alert_triggered = report["drift_score"] > DRIFT_ALERT_THRESHOLD

        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.drift_reports
                        (model_version_id, drift_score, feature_drifts, alert_triggered)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        deployment["deployment_id"],
                        report["drift_score"],
                        json.dumps(report["feature_drifts"]),
                        alert_triggered,
                    ),
                )

                if alert_triggered:
                    message = (
                        f"Data drift detected: global score={report['drift_score']:.3f} "
                        f"for model v{deployment['model_version']}"
                    )
                    cur.execute(
                        """
                        INSERT INTO public.alert_log (alert_type, severity, message)
                        VALUES ('DRIFT_DETECTED', 'HIGH', %s)
                        """,
                        (message,),
                    )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Dependency wiring ---
    active = fetch_active_deployment()

    # Reference comes from the MLflow artifact of the active run
    ref_feat = featurize_reference(active)

    # Production extraction runs once the active deployment is known
    active >> extract_production
    prod_feat = featurize_production(active, XComArg(extract_production))

    # Drift report waits for both featurized datasets
    ref_feat >> run_drift
    prod_feat >> run_drift

    # Persist results
    save_report_to_postgresql(active, XComArg(run_drift))


drift_detection_report()
