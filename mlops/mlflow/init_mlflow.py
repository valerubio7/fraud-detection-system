"""
Inicialización idempotente del MLflow Tracking Server.

Uso:
    python mlops/mlflow/init_mlflow.py

Variables de entorno:
    MLFLOW_TRACKING_URI  URI del servidor (default: http://localhost:5000)
"""

import os
import sys
import time
import urllib.error
import urllib.request

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "fraud-detection-v1"
MAX_RETRIES = 5
RETRY_INTERVAL = 3  # segundos

EXPERIMENT_TAGS = {
    "project": "fraud-detection-mlops",
    "team": "mlops",
    "data_version": "v1",
    "model_algorithm": "xgboost",
    "task": "binary_classification",
}


def wait_for_server(tracking_uri: str) -> None:
    health_url = f"{tracking_uri}/health"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        print(
            "MLflow no disponible"
            f" (intento {attempt}/{MAX_RETRIES})."
            f" Reintentando en {RETRY_INTERVAL}s..."
        )
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_INTERVAL)

    print(f"Error: MLflow no respondió en {tracking_uri}/health tras {MAX_RETRIES} intentos.")
    sys.exit(1)


def configure_experiment_tags(client: MlflowClient, experiment_id: str) -> None:
    for key, value in EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment_id, key, value)


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Conectando a MLflow en {tracking_uri}...")
    wait_for_server(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(
            f"Experiment '{EXPERIMENT_NAME}' already exists"
            f" (id={experiment_id}) — skipping creation."
        )
    configure_experiment_tags(client, experiment_id)
    print("Experiment tags configured.")

    print()
    print("=== MLflow inicializado ===")
    print(f"  Servidor:    {tracking_uri}")
    print(f"  Experimento: {EXPERIMENT_NAME}")
    print(f"  ID:          {experiment_id}")


if __name__ == "__main__":
    main()
