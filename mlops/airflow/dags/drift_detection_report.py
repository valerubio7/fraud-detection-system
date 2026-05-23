import logging
import os
from datetime import datetime
from pathlib import Path

import psycopg2
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models.xcom_arg import XComArg
from operators import EvidentlyReportOperator, TimescaleExtractOperator

from mlops.evidently.reference_data import load_reference_dataset
from model.utils.selected_features import SELECTED_FEATURES

logger = logging.getLogger(__name__)

DAG_ID = "drift_detection_report"
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
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgresql"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "fraud_metadata_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB", "fraud_metadata"),
    )


def _download_encoder(run_id: str) -> Path:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
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
        task_id="fetch_production_data", sql=_PRODUCTION_SQL, output_path="/tmp/drift_raw_production.parquet"
    )

    @task
    def featurize_reference(active: dict) -> str:
        ref_df = load_reference_dataset(
            run_id=active["mlflow_run_id"], tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        ref_df[SELECTED_FEATURES].to_parquet(REFERENCE_PARQUET, index=False)
        return REFERENCE_PARQUET

    @task
    def featurize_production(active: dict, raw_path: str) -> str:
        import pandas as pd

        from offline_features.featurizer import TransactionFeaturizer

        encoder_dir = _download_encoder(active["mlflow_run_id"])
        featurizer = TransactionFeaturizer(encoders_dir=encoder_dir)
        df = pd.read_parquet(raw_path).reset_index(drop=True)

        if len(df) < MIN_PRODUCTION_ROWS:
            raise AirflowSkipException(f"Insufficient production data for drift analysis (<{MIN_PRODUCTION_ROWS} rows)")

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
    def run_model_drift_task(deployment: dict) -> dict:
        import psycopg2

        from mlops.evidently.model_drift import fetch_labeled_predictions, run_model_drift_report

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgresql"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "fraud_metadata_user"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB", "fraud_metadata"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT f1_score, precision, recall FROM public.model_deployments WHERE id = %s",
                    (deployment["deployment_id"],),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        ref_metrics = {
            "f1_score": row[0] if row else None,
            "precision": row[1] if row else None,
            "recall": row[2] if row else None,
        }
        labeled_df = fetch_labeled_predictions(deployment["deployment_id"])
        result = run_model_drift_report(ref_metrics, labeled_df)
        return {
            "has_sufficient_data": result.has_sufficient_data,
            "drift_detected": result.drift_detected,
            "f1_degradation": result.f1_degradation,
            "current_f1": result.current_f1,
        }

    @task
    def export_html_reports(deployment: dict, report_id: int) -> dict:
        import pandas as pd
        from mlflow.tracking import MlflowClient

        from mlops.evidently.data_drift import run_data_drift_report_with_html
        from mlops.evidently.report_uploader import upload_report_to_mlflow
        from model.utils.selected_features import SELECTED_FEATURES

        ref_df = pd.read_parquet(REFERENCE_PARQUET)
        cur_df = pd.read_parquet(PRODUCTION_PARQUET)
        _, html_path = run_data_drift_report_with_html(ref_df, cur_df, SELECTED_FEATURES)

        client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mv = client.get_model_version(deployment["model_name"], deployment["model_version"])
        artifact_uri = upload_report_to_mlflow(
            run_id=mv.run_id,
            html_path=html_path,
            artifact_subfolder=f"drift_reports/report_{report_id}",
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        )
        return {"artifact_uri": artifact_uri, "report_id": report_id}

    @task
    def save_report_to_postgresql(deployment: dict, data_drift: dict, model_drift: dict) -> int:
        from mlops.evidently.drift_policy import evaluate_drift_action, trigger_retrain_dag
        from mlops.evidently.drift_store import DriftReportStore

        action = evaluate_drift_action(data_drift, model_drift)
        store = DriftReportStore()

        if action.alert_triggered:
            store.save_alert(
                alert_type="DRIFT_DETECTED",
                severity=action.severity,
                message=f"{action.alert_message} — model v{deployment['model_version']}",
            )

        if action.trigger_retraining:
            trigger_retrain_dag(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        report_id = store.save(
            deployment_id=deployment["deployment_id"],
            data_drift_score=data_drift["drift_score"],
            feature_drifts=data_drift["feature_drifts"],
            model_drift_detected=model_drift.get("drift_detected", False),
            model_f1_degradation=model_drift.get("f1_degradation"),
            alert_triggered=action.alert_triggered,
            remediation_action=action.remediation_action,
        )
        return report_id

    # --- Dependency wiring ---
    active = fetch_active_deployment()

    # Reference comes from the MLflow artifact of the active run
    ref_feat = featurize_reference(active)

    # Production extraction runs once the active deployment is known
    active >> extract_production
    prod_feat = featurize_production(active, XComArg(extract_production))

    # Data drift and model drift run in parallel
    ref_feat >> run_drift
    prod_feat >> run_drift
    model_drift_result = run_model_drift_task(active)

    # Persist both results, then export HTML to MLflow
    report_id = save_report_to_postgresql(active, XComArg(run_drift), model_drift_result)
    export_html_reports(active, report_id)


drift_detection_report()
