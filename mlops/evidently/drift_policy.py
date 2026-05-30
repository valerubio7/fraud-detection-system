import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CRITICAL_FEATURES: frozenset[str] = frozenset(
    {"tx_count_1h", "amount_sum_1h", "amount_ratio_vs_user_avg", "is_country_new", "seconds_since_last_tx"}
)

DRIFT_THRESHOLD_CRITICAL = float(os.getenv("DRIFT_THRESHOLD_CRITICAL", "0.20"))
DRIFT_THRESHOLD_GLOBAL = float(os.getenv("DRIFT_THRESHOLD_GLOBAL", "0.30"))
MODEL_F1_DEGRADATION_THRESHOLD = float(os.getenv("MODEL_F1_DEGRADATION_THRESHOLD", "0.05"))


@dataclass
class DriftAction:
    alert_triggered: bool
    severity: str
    alert_message: str
    trigger_retraining: bool
    remediation_action: str | None


def evaluate_drift_action(data_drift_result: dict, model_drift_result: dict) -> DriftAction:
    drift_score: float = data_drift_result.get("drift_score", 0.0)
    feature_drifts: dict = data_drift_result.get("feature_drifts", {})
    model_drift_detected: bool = bool(model_drift_result.get("drift_detected", False))

    critical_features_drifted = [
        name
        for name in CRITICAL_FEATURES
        if feature_drifts.get(name, {}).get("drift_score", 0.0) > DRIFT_THRESHOLD_CRITICAL
    ]
    any_critical = len(critical_features_drifted) > 0

    if any_critical and model_drift_detected:
        severity = "CRITICAL"
        message = (
            f"CRITICAL drift: critical features drifted={critical_features_drifted}, "
            f"model drift detected, global score={drift_score:.3f}"
        )
    elif any_critical or model_drift_detected:
        severity = "HIGH"
        if any_critical:
            message = (
                f"High drift: critical features drifted={critical_features_drifted}, global score={drift_score:.3f}"
            )
        else:
            deg = model_drift_result.get("f1_degradation")
            message = f"High drift: model F1 degradation={deg:.3f}, global score={drift_score:.3f}"
    elif drift_score > DRIFT_THRESHOLD_GLOBAL:
        severity = "WARNING"
        message = f"Global data drift score={drift_score:.3f} exceeds threshold={DRIFT_THRESHOLD_GLOBAL}"
    else:
        severity = "INFO"
        message = f"Drift within acceptable bounds: global score={drift_score:.3f}"

    alert_triggered = severity != "INFO"
    trigger_retraining = severity != "INFO"
    remediation_action = "triggered_retraining_dag" if trigger_retraining else None

    return DriftAction(
        alert_triggered=alert_triggered,
        severity=severity,
        alert_message=message,
        trigger_retraining=trigger_retraining,
        remediation_action=remediation_action,
    )


def trigger_retrain_dag(tracking_uri: str) -> bool:
    import requests

    airflow_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080")
    airflow_user = os.getenv("AIRFLOW_ADMIN_USER", "admin")
    airflow_pass = os.getenv("AIRFLOW_ADMIN_PASSWORD", "")
    url = f"{airflow_url}/api/v1/dags/retrain_fraud_model/dagRuns"
    try:
        resp = requests.post(
            url, json={"conf": {"triggered_by": "drift_detection"}}, auth=(airflow_user, airflow_pass), timeout=10
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Failed to trigger retrain_fraud_model DAG: %s", exc)
        return False
