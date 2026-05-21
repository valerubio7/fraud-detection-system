from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import psycopg2
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering.offline.featurizer import TransactionFeaturizer  # noqa: E402
from model.selected_features import SELECTED_FEATURES  # noqa: E402

MIN_F1 = 0.85
MIN_AUC_ROC = 0.90
MAX_LATENCY_P99_MS = 50.0
MIN_F1_IMPROVEMENT = 0.02


@dataclass
class GateResult:
    f1_score: float
    auc_roc: float
    latency_p99_ms: float
    f1_passed: bool
    auc_roc_passed: bool
    latency_passed: bool
    passed: bool


@dataclass
class ChampionComparisonResult:
    challenger_f1: float
    challenger_auc_roc: float
    champion_f1: float | None
    champion_auc_roc: float | None
    f1_difference: float | None
    challenger_wins: bool
    reason: str


def load_model(model_name: str, model_version: str) -> object:
    """Load a model version from MLflow Model Registry by name and version."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    return _load_model_from_uri(f"models:/{model_name}/{model_version}")


def load_champion_model(model_name: str) -> object | None:
    """Load the Production champion model, or None if no version is in Production."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()
    if not client.get_latest_versions(model_name, stages=["Production"]):
        return None
    try:
        return _load_model_from_uri(f"models:/{model_name}/Production")
    except Exception as exc:
        raise RuntimeError(
            f"Champion model exists in Production stage but failed to load: {exc}"
        ) from exc


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    featurizer = TransactionFeaturizer(encoders_dir="artifacts/model")
    return featurizer.transform(df)[SELECTED_FEATURES]


def measure_latency(model: object, X_sample: pd.DataFrame, n_repetitions: int = 10) -> float:
    """Return P99 prediction latency in milliseconds over n_repetitions batches."""
    X_batch = X_sample.iloc[: min(1000, len(X_sample))]
    timings = np.zeros(n_repetitions)
    for i in range(n_repetitions):
        start = time.perf_counter()
        model.predict(X_batch)
        timings[i] = (time.perf_counter() - start) * 1000.0
    return float(np.percentile(timings, 99))


def run_quality_gates(model_name: str, model_version: str) -> GateResult:
    """Evaluate a model version against quality gates and return the result."""
    print(f"Loading model {model_name} version {model_version} from MLflow...")
    model = load_model(model_name, model_version)

    print("Loading test data from TimescaleDB...")
    test_df = _load_test_data()
    y_test = test_df["is_fraud"].astype(int)
    print(f"Test set size: {len(test_df)} transactions")

    X_test = compute_features(test_df)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    f1 = float(f1_score(y_test, preds, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, proba))
    latency_p99 = measure_latency(model, X_test)

    result = GateResult(
        f1_score=f1,
        auc_roc=auc_roc,
        latency_p99_ms=latency_p99,
        f1_passed=f1 >= MIN_F1,
        auc_roc_passed=auc_roc >= MIN_AUC_ROC,
        latency_passed=latency_p99 <= MAX_LATENCY_P99_MS,
        passed=f1 >= MIN_F1 and auc_roc >= MIN_AUC_ROC and latency_p99 <= MAX_LATENCY_P99_MS,
    )

    _log_gate_metrics_to_mlflow(model_name, model_version, result)
    _print_gate_summary(result)
    _update_quality_gate_tag(model_name, model_version, result)
    return result


def compare_challenger_vs_champion(
    challenger_name: str, challenger_version: str
) -> ChampionComparisonResult:
    """Compare a challenger model against the Production champion on the test set."""
    print("Loading champion model from Production stage...")
    champion = load_champion_model(challenger_name)

    if champion is None:
        result = ChampionComparisonResult(
            challenger_f1=0.0,
            challenger_auc_roc=0.0,
            champion_f1=None,
            champion_auc_roc=None,
            f1_difference=None,
            challenger_wins=True,
            reason="No champion model in Production stage — challenger wins by default",
        )
        _print_comparison_summary(result)
        return result

    print("Loading challenger model...")
    challenger = load_model(challenger_name, challenger_version)

    print("Loading test data for comparison...")
    test_df = _load_test_data()
    y_test = test_df["is_fraud"].astype(int)
    print(f"Test set size: {len(test_df)} transactions")

    X_test = compute_features(test_df)

    challenger_proba = challenger.predict_proba(X_test)[:, 1]
    challenger_f1 = float(f1_score(y_test, (challenger_proba >= 0.5).astype(int), zero_division=0))
    challenger_auc = float(roc_auc_score(y_test, challenger_proba))

    champion_proba = champion.predict_proba(X_test)[:, 1]
    champion_f1 = float(f1_score(y_test, (champion_proba >= 0.5).astype(int), zero_division=0))
    champion_auc = float(roc_auc_score(y_test, champion_proba))

    f1_diff = challenger_f1 - champion_f1
    challenger_wins = challenger_f1 > champion_f1 + MIN_F1_IMPROVEMENT
    reason = (
        f"Challenger F1 ({challenger_f1:.4f}) exceeds champion F1 ({champion_f1:.4f}) "
        f"by more than {MIN_F1_IMPROVEMENT:.2f} (diff: {f1_diff:+.4f})"
        if challenger_wins
        else f"Challenger F1 ({challenger_f1:.4f}) does not exceed champion F1 "
        f"({champion_f1:.4f}) by at least {MIN_F1_IMPROVEMENT:.2f} (diff: {f1_diff:+.4f})"
    )

    result = ChampionComparisonResult(
        challenger_f1=challenger_f1,
        challenger_auc_roc=challenger_auc,
        champion_f1=champion_f1,
        champion_auc_roc=champion_auc,
        f1_difference=f1_diff,
        challenger_wins=challenger_wins,
        reason=reason,
    )
    _print_comparison_summary(result)
    return result


def _load_test_data() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.getenv("TIMESCALE_HOST", "timescaledb"),
        port=int(os.getenv("TIMESCALE_PORT", "5432")),
        user=os.getenv("TIMESCALE_USER", "fraud_timeseries_user"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB", "fraud_transactions_timeseries"),
    )
    query = """
        SELECT transaction_id, user_id, merchant_id, merchant_category,
               amount, country, device_type, ip_hash, timestamp, is_fraud
        FROM public.transactions
        WHERE is_fraud IS NOT NULL
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} labeled transactions found. At least 100 are required.")
    split_idx = int(len(df) * 0.8)
    return df.iloc[split_idx:].reset_index(drop=True)


def _load_model_from_uri(uri: str) -> object:
    import xgboost as xgb

    saved = getattr(xgb.XGBModel, "_estimator_type", None)
    xgb.XGBModel._estimator_type = "classifier"
    try:
        model = mlflow.xgboost.load_model(uri)
    finally:
        if saved is not None:
            xgb.XGBModel._estimator_type = saved
        else:
            del xgb.XGBModel._estimator_type

    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    if not hasattr(model, "n_classes_"):
        model.n_classes_ = 2
    return model


def _update_quality_gate_tag(model_name: str, model_version: str, result: GateResult) -> None:
    try:
        client = MlflowClient()
        tag_value = "passed" if result.passed else "failed"
        client.set_model_version_tag(model_name, model_version, "quality_gates", tag_value)
    except Exception as exc:
        print(f"Warning: Failed to update quality_gates tag: {exc}")


def _log_gate_metrics_to_mlflow(model_name: str, model_version: str, result: GateResult) -> None:
    try:
        client = MlflowClient()
        run_id = client.get_model_version(model_name, model_version).run_id
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(
                    {
                        "quality_gate_f1_score": result.f1_score,
                        "quality_gate_auc_roc": result.auc_roc,
                        "quality_gate_latency_p99_ms": result.latency_p99_ms,
                        "quality_gate_passed": float(result.passed),
                    }
                )
            print(f"Metrics logged to MLflow run {run_id}")
        else:
            print("Warning: Could not find MLflow run for this model version.")
    except Exception as exc:
        print(f"Warning: Failed to log metrics to MLflow: {exc}")


def _print_gate_summary(result: GateResult) -> None:
    print()
    print("=" * 50)
    print("QUALITY GATES REPORT")
    print("=" * 50)
    for label, value, passed, threshold in [
        ("F1-score (fraud)", result.f1_score, result.f1_passed, f">= {MIN_F1}"),
        ("AUC-ROC", result.auc_roc, result.auc_roc_passed, f">= {MIN_AUC_ROC}"),
        (
            "Latency P99 (ms)",
            result.latency_p99_ms,
            result.latency_passed,
            f"<= {MAX_LATENCY_P99_MS}",
        ),
    ]:
        print(f"  {label:<22} {value:.4f}  {'PASS' if passed else 'FAIL':4s}  ({threshold})")
    print("-" * 50)
    print(f"  {'Overall:':<22} {'PASS' if result.passed else 'FAIL'}")
    print("=" * 50)


def _print_comparison_summary(result: ChampionComparisonResult) -> None:
    print()
    print("=" * 50)
    print("CHALLENGER vs CHAMPION COMPARISON")
    print("=" * 50)
    if result.champion_f1 is None:
        print("  No champion model in Production stage.")
        print("  Challenger wins by default.")
        print("=" * 50)
        return
    print(f"  {'Metric':<22} {'Challenger':>10} {'Champion':>10} {'Diff':>10}")
    print(f"  {'------':<22} {'----------':>10} {'----------':>10} {'------':>10}")
    print(
        f"  {'F1-score (fraud)':<22} {result.challenger_f1:>10.4f} "
        f"{result.champion_f1:>10.4f} {result.f1_difference:>+10.4f}"
    )
    print(
        f"  {'AUC-ROC':<22} {result.challenger_auc_roc:>10.4f} "
        f"{result.champion_auc_roc:>10.4f} {'':>10}"
    )
    print("-" * 50)
    print(f"  {'Verdict:':<22} {'WIN' if result.challenger_wins else 'LOSE'}")
    print(f"  Reason: {result.reason}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate quality gates and compare challenger vs champion."
    )
    parser.add_argument("--model-name", required=True, help="MLflow Model Registry name.")
    parser.add_argument("--model-version", required=True, help="Model version number.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the model against the Production champion after quality gates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_quality_gates(args.model_name, args.model_version)
    if not result.passed:
        sys.exit(1)

    if args.compare:
        comparison = compare_challenger_vs_champion(args.model_name, args.model_version)
        if not comparison.challenger_wins:
            print(f"\nChallenger rejected: {comparison.reason}")
            sys.exit(1)
        print("\nChallenger passed all checks — ready for promotion.")


if __name__ == "__main__":
    main()
