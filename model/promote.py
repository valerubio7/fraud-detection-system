"""Promote a model version to Production in MLflow Registry and record the deployment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg2
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


@dataclass
class PromotionResult:
    """Result of a model promotion to Production."""

    model_deployment_id: int
    mlflow_run_id: str
    model_version: str
    stage: str
    promoted_at: str


def promote_to_production(model_name: str, model_version: str) -> PromotionResult:
    """Promote a model version from Staging to Production.

    Args:
        model_name: Name of the model in MLflow Model Registry.
        model_version: Version number to promote.

    Returns:
        PromotionResult with deployment details.
    """
    mlflow_settings = config.mlflow_settings
    client = MlflowClient(tracking_uri=mlflow_settings.tracking_uri)

    # Step 1 — verify model exists and is in Staging
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

    run_id = mv.run_id
    print(f"Fetching run {run_id} metadata from MLflow...")
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params_dict = run.data.params

    f1_score = metrics.get("f1_score")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    auc_roc = metrics.get("auc_roc")

    raw_from = params_dict.get("training_data_from")
    raw_to = params_dict.get("training_data_to")

    run_start = datetime.fromtimestamp(run.info.start_time / 1000.0, tz=UTC)
    promoted_at = datetime.now(UTC)

    def _parse_ts(raw: str | None, fallback: datetime) -> datetime:
        if not raw:
            return fallback
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return fallback

    training_data_from = _parse_ts(raw_from, run_start)
    training_data_to = _parse_ts(raw_to, promoted_at)

    # Step 2-3 — PostgreSQL transaction (insert + activate)
    postgres_settings = config.postgres_settings
    conn = psycopg2.connect(
        host=postgres_settings.host,
        port=postgres_settings.port,
        user=postgres_settings.user,
        password=postgres_settings.password,
        dbname=postgres_settings.db,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM public.model_deployments WHERE mlflow_run_id = %s",
                (run_id,),
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
                    VALUES (%s, %s, %s, %s, FALSE,
                            %s, %s, %s, %s,
                            %s, %s)
                    RETURNING id
                    """,
                    (
                        model_name,
                        model_version,
                        run_id,
                        promoted_at,
                        f1_score,
                        precision,
                        recall,
                        auc_roc,
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

    # Step 4 — Transition MLflow stage (only after DB transaction succeeded)
    try:
        print(f"Transitioning {model_name} v{model_version} to Production...")
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

    result = PromotionResult(
        model_deployment_id=deployment_id,
        mlflow_run_id=run_id,
        model_version=model_version,
        stage="Production",
        promoted_at=promoted_at.isoformat(),
    )

    # Step 5 — Print summary
    print()
    print("=" * 50)
    print("MODEL PROMOTION REPORT")
    print("=" * 50)
    print(f"  Model:              {model_name}")
    print(f"  Version:            {model_version}")
    print("  Stage:              Production")
    print(f"  MLflow Run ID:      {run_id}")
    print(f"  Deployment DB ID:   {deployment_id}")
    print(f"  Promoted at:        {promoted_at.isoformat()}")
    print(f"  F1-score:           {f1_score if f1_score is not None else 'N/A'}")
    print(f"  AUC-ROC:            {auc_roc if auc_roc is not None else 'N/A'}")
    print("=" * 50)

    return result


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
