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

EXPERIMENT_NAME = "fraud-detection-v1"
MAX_RETRIES = 5
RETRY_INTERVAL = 3  # segundos


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


def ensure_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        print(f"Experiment '{name}' already exists — skipping creation.")
        return experiment.experiment_id

    experiment_id = mlflow.create_experiment(name)
    print(f"Experimento '{name}' creado con ID {experiment_id}.")
    return experiment_id


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Conectando a MLflow en {tracking_uri}...")
    wait_for_server(tracking_uri)

    experiment_id = ensure_experiment(EXPERIMENT_NAME)

    print()
    print("=== MLflow inicializado ===")
    print(f"  Servidor:    {tracking_uri}")
    print(f"  Experimento: {EXPERIMENT_NAME}")
    print(f"  ID:          {experiment_id}")


if __name__ == "__main__":
    main()
