import logging
import os
import time
import urllib.error
import urllib.request

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "fraud-detection-v1"
MODEL_NAME = "FraudDetectionModel"
MAX_RETRIES = 5
RETRY_INTERVAL = 3  # seconds

EXPERIMENT_TAGS = {
    "project": "fraud-detection-mlops",
    "team": "mlops",
    "data_version": "v1",
    "model_algorithm": "xgboost",
    "task": "binary_classification",
}


def wait_for_server(tracking_uri: str) -> None:
    health_url = f"{tracking_uri.rstrip('/')}/health"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        logger.warning("MLflow not available (attempt %d/%d). Retrying in %ds...", attempt, MAX_RETRIES, RETRY_INTERVAL)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_INTERVAL)

    raise RuntimeError(f"MLflow did not respond at {health_url} after {MAX_RETRIES} attempts.")


def configure_experiment_tags(client: MlflowClient, experiment_id: str) -> None:
    for key, value in EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment_id, key, value)


def register_model_metadata(client: MlflowClient, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
        logger.info("Registered model '%s' created.", model_name)
    except MlflowException:
        pass  # already exists

    try:
        client.update_registered_model(
            name=model_name,
            description=(
                "XGBoost binary classifier for real-time fraud detection. "
                "Trained on TimescaleDB transaction history with 16 engineered features. "
                "Lifecycle: None → Staging (train.py)"
                " → Production (promote.py after quality gates) → Archived."
            ),
        )
        client.set_registered_model_tag(model_name, "task", "binary_classification")
        client.set_registered_model_tag(model_name, "algorithm", "xgboost")
        client.set_registered_model_tag(model_name, "input_features", "16")
        client.set_registered_model_tag(model_name, "feature_selection", "importance_threshold+correlation")
        client.set_registered_model_tag(model_name, "serving_endpoint", "POST /predict")
    except MlflowException as exc:
        logger.error("Failed to update model metadata for '%s': %s", model_name, exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info("Connecting to MLflow at %s...", tracking_uri)
    wait_for_server(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info("Created experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Experiment '%s' already exists (id=%s) — skipping creation.", EXPERIMENT_NAME, experiment_id)

    configure_experiment_tags(client, experiment_id)
    logger.info("Experiment tags configured.")

    register_model_metadata(client, MODEL_NAME)

    logger.info("=== MLflow initialized ===")
    logger.info("  Server:     %s", tracking_uri)
    logger.info("  Experiment: %s", EXPERIMENT_NAME)
    logger.info("  ID:         %s", experiment_id)


if __name__ == "__main__":
    main()
