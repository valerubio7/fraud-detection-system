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
import matplotlib.pyplot as plt


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
