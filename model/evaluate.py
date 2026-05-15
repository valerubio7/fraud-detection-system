"""Evaluation utilities for the fraud detection model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import argparse
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
import psycopg2
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from feature_engineering.offline.featurizer import TransactionFeaturizer  # noqa: E402
from model.features import SELECTED_FEATURES  # noqa: E402

# ---------------------------------------------------------------------------
# Quality gates — hardcoded thresholds (modify here to adjust)
# ---------------------------------------------------------------------------
MIN_F1 = 0.85
MIN_AUC_ROC = 0.90
MAX_LATENCY_P99_MS = 50.0


def compute_threshold_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, list[float]]:
    metrics = {
        "thresholds": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        metrics["thresholds"].append(float(threshold))
        metrics["precision"].append(float(precision_score(y_true, preds, zero_division=0)))
        metrics["recall"].append(float(recall_score(y_true, preds, zero_division=0)))
        metrics["f1"].append(float(f1_score(y_true, preds, zero_division=0)))
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
    *,
    cost_false_negative: float = 100.0,
    cost_false_positive: float = 5.0,
) -> tuple[float, dict[str, list[float]]]:
    metrics = compute_threshold_metrics(y_true, proba, thresholds)
    costs = []
    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        fn = int(((y_true == 1) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        costs.append(fn * cost_false_negative + fp * cost_false_positive)
    best_index = int(np.argmin(costs))
    return float(metrics["thresholds"][best_index]), metrics


def evaluate_model(
    model,
    X_test,
    y_test,
    *,
    threshold: float = 0.5,
    cost_false_negative: float = 100.0,
    cost_false_positive: float = 5.0,
) -> dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))

    matrix = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    total_cost = float(fn * cost_false_negative + fp * cost_false_positive)
    cost_per_transaction = float(total_cost / len(y_test)) if len(y_test) else 0.0
    fraud_detected_pct = float(tp / (tp + fn)) if (tp + fn) else 0.0
    legitimate_blocked_pct = float(fp / (fp + tn)) if (fp + tn) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "fpr": fpr,
        "fnr": fnr,
        "threshold": float(threshold),
        "cost_false_negative": float(cost_false_negative),
        "cost_false_positive": float(cost_false_positive),
        "total_cost": total_cost,
        "cost_per_transaction": cost_per_transaction,
        "fraud_detected_pct": fraud_detected_pct,
        "legitimate_blocked_pct": legitimate_blocked_pct,
    }


def save_confusion_matrix_plot(
    confusion: dict[str, int],
    output_path: Path,
) -> None:
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Legitima", "Fraude"])
    ax.set_yticks([0, 1], labels=["Legitima", "Fraude"])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, int(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_roc_curve_plot(
    y_true: np.ndarray,
    proba: np.ndarray,
    output_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_value = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pr_curve_plot(
    y_true: np.ndarray,
    proba: np.ndarray,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true)) if len(y_true) else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.hlines(
        baseline,
        xmin=0,
        xmax=1,
        colors="gray",
        linestyles="--",
        label="Baseline",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_feature_importance_plot(
    model,
    feature_names: list[str],
    output_path: Path,
    *,
    top_n: int = 20,
) -> None:
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    resolved = []
    for name, score in scores.items():
        resolved_name = _resolve_feature_name(name, feature_names)
        resolved.append((resolved_name, float(score)))

    resolved.sort(key=lambda item: item[1], reverse=True)
    top = resolved[:top_n]

    fig, ax = plt.subplots(figsize=(7, 5))
    if not top:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [item[0] for item in reversed(top)]
        values = [item[1] for item in reversed(top)]
        ax.barh(labels, values, color="#1f77b4")
        ax.set_xlabel("Gain")
        ax.set_title("Top Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_threshold_analysis_plot(
    threshold_metrics: dict[str, list[float]],
    optimal_threshold: float,
    output_path: Path,
) -> None:
    thresholds = threshold_metrics["thresholds"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, threshold_metrics["f1"], label="F1")
    ax.plot(thresholds, threshold_metrics["precision"], label="Precision")
    ax.plot(thresholds, threshold_metrics["recall"], label="Recall")
    ax.axvline(optimal_threshold, color="red", linestyle="--", label="Optimal")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _resolve_feature_name(name: str, feature_names: list[str]) -> str:
    if name in feature_names:
        return name
    if name.startswith("f") and name[1:].isdigit():
        index = int(name[1:])
        if 0 <= index < len(feature_names):
            return feature_names[index]
    return name


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Structured result of quality gate evaluation."""

    f1_score: float
    auc_roc: float
    latency_p99_ms: float
    f1_passed: bool
    auc_roc_passed: bool
    latency_passed: bool
    passed: bool


def load_test_data() -> pd.DataFrame:
    """Load the last 20% temporal split of labeled transactions from TimescaleDB."""
    settings = config.timescaledb_settings
    conn = psycopg2.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        dbname=settings.db,
    )
    query = """
        SELECT
            transaction_id, user_id, merchant_id, merchant_category,
            amount, country, device_type, ip_hash, timestamp, is_fraud
        FROM public.transactions
        WHERE is_fraud IS NOT NULL
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} labeled transactions found. At least 100 are required for evaluation."
        )

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return test_df


def load_model(model_name: str, model_version: str):
    """Load an XGBoost model from MLflow Model Registry."""
    mlflow_settings = config.mlflow_settings
    mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
    uri = f"models:/{model_name}/{model_version}"

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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TransactionFeaturizer.transform() and select model features."""
    featurizer = TransactionFeaturizer(encoders_dir="artifacts/model")
    X = featurizer.transform(df)
    X = X[SELECTED_FEATURES]
    return X


def measure_latency(model, X_sample: pd.DataFrame, n_repetitions: int = 10) -> float:
    """Measure P99 prediction latency in milliseconds over n_repetitions batches."""
    n = min(1000, len(X_sample))
    X_batch = X_sample.iloc[:n]
    timings = np.zeros(n_repetitions)
    for i in range(n_repetitions):
        start = time.perf_counter()
        model.predict(X_batch)
        timings[i] = (time.perf_counter() - start) * 1000.0
    return float(np.percentile(timings, 99))


def run_quality_gates(model_name: str, model_version: str) -> GateResult:
    """Evaluate a model version and reject it if quality gates are not met.

    Args:
        model_name: Name of the model in MLflow Model Registry.
        model_version: Version number of the model to evaluate.

    Returns:
        GateResult with computed metrics and pass/fail per gate.
    """
    print(f"Loading model {model_name} version {model_version} from MLflow...")
    model = load_model(model_name, model_version)

    print("Loading test data from TimescaleDB...")
    test_df = load_test_data()
    y_test = test_df["is_fraud"].astype(int)
    print(f"Test set size: {len(test_df)} transactions")

    print("Computing features...")
    X_test = compute_features(test_df)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    f1 = float(f1_score(y_test, preds, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, proba))

    latency_p99 = measure_latency(model, X_test)

    f1_passed = f1 >= MIN_F1
    auc_roc_passed = auc_roc >= MIN_AUC_ROC
    latency_passed = latency_p99 <= MAX_LATENCY_P99_MS
    overall_passed = f1_passed and auc_roc_passed and latency_passed

    result = GateResult(
        f1_score=f1,
        auc_roc=auc_roc,
        latency_p99_ms=latency_p99,
        f1_passed=f1_passed,
        auc_roc_passed=auc_roc_passed,
        latency_passed=latency_passed,
        passed=overall_passed,
    )

    _log_metrics_to_mlflow(model_name, model_version, result)
    _print_summary(result)

    return result


def _log_metrics_to_mlflow(model_name: str, model_version: str, result: GateResult) -> None:
    """Log quality gate metrics to the MLflow run associated with the model version."""
    try:
        client = MlflowClient()
        mv_details = client.get_model_version(model_name, model_version)
        target_run_id = mv_details.run_id

        if target_run_id:
            with mlflow.start_run(run_id=target_run_id):
                mlflow.log_metrics(
                    {
                        "quality_gate_f1_score": result.f1_score,
                        "quality_gate_auc_roc": result.auc_roc,
                        "quality_gate_latency_p99_ms": result.latency_p99_ms,
                        "quality_gate_passed": float(result.passed),
                    }
                )
            print(f"Metrics logged to MLflow run {target_run_id}")
        else:
            print("Warning: Could not find MLflow run for this model version.")
    except Exception as exc:
        print(f"Warning: Failed to log metrics to MLflow: {exc}")


def _print_summary(result: GateResult) -> None:
    """Print a clear summary of quality gate results."""
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
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<22} {value:.4f}  {status:4s}  ({threshold})")
    print("-" * 50)
    print(f"  {'Overall:':<22} {'PASS' if result.passed else 'FAIL'}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model quality gates.")
    parser.add_argument("--model-name", required=True, help="MLflow Model Registry name.")
    parser.add_argument("--model-version", required=True, help="Model version number.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_quality_gates(args.model_name, args.model_version)
    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
