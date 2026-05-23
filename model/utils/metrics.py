from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {"thresholds": [], "precision": [], "recall": [], "f1": []}
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
    model: Any,
    X_test: Any,
    y_test: Any,
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
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

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
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
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


__all__ = ["compute_threshold_metrics", "evaluate_model", "find_optimal_threshold"]
