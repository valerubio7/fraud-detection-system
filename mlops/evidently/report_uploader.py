import logging
import os

logger = logging.getLogger(__name__)


def upload_report_to_mlflow(run_id: str, html_path: str, artifact_subfolder: str, tracking_uri: str) -> str | None:
    try:
        from mlflow.tracking import MlflowClient

        MlflowClient(tracking_uri=tracking_uri).log_artifact(run_id, html_path, artifact_path=artifact_subfolder)
        return f"runs:/{run_id}/{artifact_subfolder}/{os.path.basename(html_path)}"
    except Exception:
        logger.warning("Failed to upload HTML report to MLflow (run_id=%s, path=%s)", run_id, html_path, exc_info=True)
        return None
