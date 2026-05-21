from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg2
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class PromotionResult:
    model_deployment_id: int
    mlflow_run_id: str
    model_version: str
    stage: str
    promoted_at: str


def promote_to_production(model_name: str, model_version: str) -> PromotionResult:
    """Promote a model version from Staging to Production."""
    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    mv, run_id = _verify_staging_version(client, model_name, model_version)
    run_metrics, run_params, run_start = _fetch_run_metadata(client, run_id)

    promoted_at = datetime.now(UTC)
    training_data_from = _parse_timestamp_with_fallback(
        run_params.get("training_data_from"), run_start
    )
    training_data_to = _parse_timestamp_with_fallback(
        run_params.get("training_data_to"), promoted_at
    )

    deployment_id = _record_deployment_in_db(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        promoted_at=promoted_at,
        metrics=run_metrics,
        training_data_from=training_data_from,
        training_data_to=training_data_to,
    )

    _transition_mlflow_stage(client, model_name, model_version)

    result = PromotionResult(
        model_deployment_id=deployment_id,
        mlflow_run_id=run_id,
        model_version=model_version,
        stage="Production",
        promoted_at=promoted_at.isoformat(),
    )
    _print_promotion_summary(model_name, run_id, deployment_id, promoted_at, run_metrics, result)
    return result


def _verify_staging_version(
    client: MlflowClient, model_name: str, model_version: str
) -> tuple[object, str]:
    print(f"Checking model {model_name} version {model_version} in MLflow...")
    try:
        mv = client.get_model_version(model_name, model_version)
    except Exception as exc:
        print(f"Error: Model {model_name} version {model_version} not found in MLflow: {exc}")
        sys.exit(1)

    if mv.current_stage != "Staging":
        print(
            f"Error: Model {model_name} v{model_version} is in stage "
            f"'{mv.current_stage}', expected 'Staging'."
        )
        sys.exit(1)

    return mv, mv.run_id


def _fetch_run_metadata(client: MlflowClient, run_id: str) -> tuple[dict, dict, datetime]:
    print(f"Fetching run {run_id} metadata from MLflow...")
    run = client.get_run(run_id)
    run_start = datetime.fromtimestamp(run.info.start_time / 1000.0, tz=UTC)
    return run.data.metrics, run.data.params, run_start


def _record_deployment_in_db(
    *,
    run_id: str,
    model_name: str,
    model_version: str,
    promoted_at: datetime,
    metrics: dict,
    training_data_from: datetime,
    training_data_to: datetime,
) -> int:
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
                "SELECT id FROM public.model_deployments WHERE mlflow_run_id = %s", (run_id,)
            )
            row = cur.fetchone()

            if row is not None:
                deployment_id = row[0]
                print(
                    f"Deployment record already exists for run {run_id} "
                    f"(id={deployment_id}), reusing."
                )
            else:
                cur.execute(
                    """
                    INSERT INTO public.model_deployments
                        (model_name, version, mlflow_run_id, created_at, is_active,
                         f1_score, precision, recall, auc_roc,
                         training_data_from, training_data_to)
                    VALUES (%s, %s, %s, %s, FALSE, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        model_name,
                        model_version,
                        run_id,
                        promoted_at,
                        metrics.get("f1_score"),
                        metrics.get("precision"),
                        metrics.get("recall"),
                        metrics.get("auc_roc"),
                        training_data_from,
                        training_data_to,
                    ),
                )
                deployment_id = cur.fetchone()[0]
                print(f"Inserted deployment record id={deployment_id}.")

            print(f"Calling activate_model_version({deployment_id})...")
            cur.execute("SELECT public.activate_model_version(%s)", (deployment_id,))

        conn.commit()
        print("PostgreSQL transaction committed.")
    except Exception as exc:
        conn.rollback()
        print(f"Error during PostgreSQL transaction: {exc}")
        sys.exit(1)
    finally:
        conn.close()

    return deployment_id


def _transition_mlflow_stage(client: MlflowClient, model_name: str, model_version: str) -> None:
    print(f"Transitioning {model_name} v{model_version} to Production...")
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print("Stage transition complete.")
    except Exception as exc:
        print(f"Error: MLflow stage transition failed: {exc}")
        sys.exit(1)


def _print_promotion_summary(
    model_name: str,
    run_id: str,
    deployment_id: int,
    promoted_at: datetime,
    metrics: dict,
    result: PromotionResult,
) -> None:
    print()
    print("=" * 50)
    print("MODEL PROMOTION REPORT")
    print("=" * 50)
    print(f"  Model:              {model_name}")
    print(f"  Version:            {result.model_version}")
    print("  Stage:              Production")
    print(f"  MLflow Run ID:      {run_id}")
    print(f"  Deployment DB ID:   {deployment_id}")
    print(f"  Promoted at:        {promoted_at.isoformat()}")
    print(f"  F1-score:           {metrics.get('f1_score', 'N/A')}")
    print(f"  AUC-ROC:            {metrics.get('auc_roc', 'N/A')}")
    print("=" * 50)


def _parse_timestamp_with_fallback(raw: str | None, fallback: datetime) -> datetime:
    if not raw:
        return fallback
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model version to Production.")
    parser.add_argument("--model-name", required=True, help="MLflow Model Registry name.")
    parser.add_argument("--model-version", required=True, help="Model version number.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    promote_to_production(args.model_name, args.model_version)


if __name__ == "__main__":
    main()
