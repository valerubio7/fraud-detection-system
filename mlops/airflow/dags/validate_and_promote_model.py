import logging
import os
from datetime import datetime

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

DAG_ID = "validate_and_promote_model"


@dag(
    dag_id=DAG_ID,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "promotion"],
    default_args={"retries": 0},
)
def validate_and_promote_model() -> None:
    @task
    def extract_model_params() -> dict:
        from airflow.operators.python import get_current_context

        context = get_current_context()
        conf = context["dag_run"].conf or {}
        model_name = conf.get("model_name")
        model_version = conf.get("model_version")
        if not model_name or not model_version:
            raise ValueError("dag_run.conf must include model_name and model_version")
        return {"model_name": str(model_name), "model_version": str(model_version)}

    @task
    def run_quality_gates_task(model_params: dict) -> dict:
        from model.pipeline.evaluate import run_quality_gates

        result = run_quality_gates(model_params["model_name"], model_params["model_version"])
        return {
            "passed": result.passed,
            "f1": result.f1_score,
            "auc_roc": result.auc_roc,
            "latency_p99_ms": result.latency_p99_ms,
        }

    @task
    def compare_with_champion(model_params: dict, gate_result: dict) -> dict:
        if not gate_result["passed"]:
            raise AirflowSkipException("Quality gates failed — skipping champion comparison")

        from model.pipeline.evaluate import compare_challenger_vs_champion

        result = compare_challenger_vs_champion(model_params["model_name"], model_params["model_version"])
        return {
            "challenger_wins": result.challenger_wins,
            "reason": result.reason,
            "f1_difference": result.f1_difference,
        }

    @task
    def promote_to_production_task(model_params: dict, comparison: dict) -> dict:
        if not comparison["challenger_wins"]:
            raise AirflowSkipException(comparison["reason"])

        from model.pipeline.promote import promote_to_production

        result = promote_to_production(model_params["model_name"], model_params["model_version"])
        return {"deployment_id": result.model_deployment_id, "promoted_at": result.promoted_at}

    @task(trigger_rule=TriggerRule.ONE_FAILED)
    def archive_rejected_version(model_params: dict, gate_result: dict) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()
        version = model_params["model_version"]
        client.transition_model_version_stage(name=model_params["model_name"], version=version, stage="Archived")
        logger.warning("Model v%s archived after failing quality gates or champion comparison", version)

    data = extract_model_params()
    gates = run_quality_gates_task(data)
    comparison = compare_with_champion(data, gates)
    promotion = promote_to_production_task(data, comparison)

    archive = archive_rejected_version(data, gates)
    comparison >> archive
    promotion >> archive


validate_and_promote_model()
