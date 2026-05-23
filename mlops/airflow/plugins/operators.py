import os
from typing import Any

from airflow.models import BaseOperator


class TimescaleExtractOperator(BaseOperator):
    template_fields = ("sql", "output_path")

    def __init__(
        self, sql: str, output_path: str, conn_settings_fn: str = "timescaledb_settings", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.sql = sql
        self.output_path = output_path
        self.conn_settings_fn = conn_settings_fn

    def execute(self, context: dict) -> str:
        import pandas as pd
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("TIMESCALE_HOST", "timescaledb"),
            port=int(os.getenv("TIMESCALE_PORT", "5432")),
            user=os.getenv("TIMESCALE_USER", "fraud_timeseries_user"),
            password=os.getenv("TIMESCALE_PASSWORD"),
            dbname=os.getenv("TIMESCALE_DB", "fraud_transactions_timeseries"),
        )
        try:
            df = pd.read_sql(self.sql, conn)
        finally:
            conn.close()

        df.to_parquet(self.output_path, index=False)
        self.log.info("Extracted %d rows to %s", len(df), self.output_path)
        return self.output_path


class MLflowRegisterModelOperator(BaseOperator):
    def __init__(
        self,
        model_name: str,
        model_version_xcom_task_id: str,
        model_version_xcom_key: str = "model_version",
        target_stage: str = "Staging",
        archive_existing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_version_xcom_task_id = model_version_xcom_task_id
        self.model_version_xcom_key = model_version_xcom_key
        self.target_stage = target_stage
        self.archive_existing = archive_existing

    def execute(self, context: dict) -> str:
        from mlflow.tracking import MlflowClient

        model_version = context["ti"].xcom_pull(
            task_ids=self.model_version_xcom_task_id, key=self.model_version_xcom_key
        )
        client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client.transition_model_version_stage(
            name=self.model_name,
            version=str(model_version),
            stage=self.target_stage,
            archive_existing_versions=self.archive_existing,
        )
        self.log.info("Model %s v%s transitioned to %s", self.model_name, model_version, self.target_stage)
        return str(model_version)


class EvidentlyReportOperator(BaseOperator):
    def __init__(
        self, reference_path_xcom_task_id: str, current_path_xcom_task_id: str, columns: list[str], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.reference_path_xcom_task_id = reference_path_xcom_task_id
        self.current_path_xcom_task_id = current_path_xcom_task_id
        self.columns = columns

    def execute(self, context: dict) -> dict:
        import pandas as pd

        from mlops.evidently.data_drift import run_data_drift_report

        ref_path = context["ti"].xcom_pull(task_ids=self.reference_path_xcom_task_id)
        cur_path = context["ti"].xcom_pull(task_ids=self.current_path_xcom_task_id)

        ref_df = pd.read_parquet(ref_path)
        cur_df = pd.read_parquet(cur_path)

        result = run_data_drift_report(ref_df, cur_df, self.columns)
        return {
            "drift_score": result.drift_share,
            "dataset_drift": result.dataset_drift,
            "feature_drifts": {
                name: {"drift_detected": fr.drift_detected, "drift_score": fr.drift_score}
                for name, fr in result.feature_results.items()
            },
        }
