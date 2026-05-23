import os


class DriftReportStore:
    def save(
        self,
        deployment_id: int,
        data_drift_score: float,
        feature_drifts: dict,
        model_drift_detected: bool = False,
        model_f1_degradation: float | None = None,
        alert_triggered: bool = False,
        remediation_action: str | None = None,
    ) -> int:
        import json

        import psycopg2

        combined_feature_drifts = {
            "data_drift": feature_drifts,
            "model_drift": {"detected": model_drift_detected, "f1_degradation": model_f1_degradation},
        }
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
                    """
                    INSERT INTO public.drift_reports
                        (model_version_id, drift_score, feature_drifts,
                         alert_triggered, remediation_action)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        deployment_id,
                        data_drift_score,
                        json.dumps(combined_feature_drifts),
                        alert_triggered,
                        remediation_action,
                    ),
                )
                report_id: int = cur.fetchone()[0]
            conn.commit()
        finally:
            conn.close()
        return report_id

    def save_alert(self, alert_type: str, severity: str, message: str) -> None:
        import psycopg2

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
                    "INSERT INTO public.alert_log (alert_type, severity, message) VALUES (%s, %s, %s)",
                    (alert_type, severity, message),
                )
            conn.commit()
        finally:
            conn.close()
