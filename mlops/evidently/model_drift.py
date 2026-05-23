import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelDriftResult:
    has_sufficient_data: bool
    reference_f1: float | None
    current_f1: float | None
    reference_precision: float | None
    current_precision: float | None
    reference_recall: float | None
    current_recall: float | None
    f1_degradation: float | None
    drift_detected: bool


_MIN_LABELED_ROWS = 50


def fetch_labeled_predictions(deployment_id: int, lookback_days: int = 7) -> pd.DataFrame:
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
                """
                SELECT prediction_score, prediction_label, actual_label
                FROM public.predictions_history
                WHERE model_version_id = %s
                  AND actual_label IS NOT NULL
                  AND timestamp >= NOW() - INTERVAL '%s days'
                """,
                (deployment_id, lookback_days),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["prediction_proba", "prediction", "target"])

    df = pd.DataFrame(rows, columns=["prediction_proba", "prediction", "target"])
    df["target"] = df["target"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["prediction_proba"] = df["prediction_proba"].astype(float)
    return df


def run_model_drift_report(reference_metrics: dict, labeled_predictions: pd.DataFrame) -> ModelDriftResult:
    if len(labeled_predictions) < _MIN_LABELED_ROWS:
        return ModelDriftResult(
            has_sufficient_data=False,
            reference_f1=None,
            current_f1=None,
            reference_precision=None,
            current_precision=None,
            reference_recall=None,
            current_recall=None,
            f1_degradation=None,
            drift_detected=False,
        )

    reference_f1 = _to_float(reference_metrics.get("f1_score"))
    reference_precision = _to_float(reference_metrics.get("precision"))
    reference_recall = _to_float(reference_metrics.get("recall"))

    y_true = labeled_predictions["target"].to_numpy()
    y_pred = labeled_predictions["prediction"].to_numpy()

    try:
        from sklearn.metrics import f1_score, precision_score, recall_score

        current_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        current_precision = float(precision_score(y_true, y_pred, zero_division=0))
        current_recall = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        current_f1 = None
        current_precision = None
        current_recall = None

    f1_degradation = (current_f1 - reference_f1) if current_f1 is not None and reference_f1 is not None else None
    drift_detected = f1_degradation is not None and f1_degradation < -0.05

    return ModelDriftResult(
        has_sufficient_data=True,
        reference_f1=reference_f1,
        current_f1=current_f1,
        reference_precision=reference_precision,
        current_precision=current_precision,
        reference_recall=reference_recall,
        current_recall=current_recall,
        f1_degradation=f1_degradation,
        drift_detected=drift_detected,
    )


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
