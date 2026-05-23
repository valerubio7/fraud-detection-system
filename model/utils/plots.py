from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def save_confusion_matrix_plot(confusion: dict[str, int], output_path: Path) -> None:
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


def save_roc_curve_plot(y_true: np.ndarray, proba: np.ndarray, output_path: Path) -> None:
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


def save_pr_curve_plot(y_true: np.ndarray, proba: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true)) if len(y_true) else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.hlines(baseline, xmin=0, xmax=1, colors="gray", linestyles="--", label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_feature_importance_plot(model: Any, feature_names: list[str], output_path: Path, *, top_n: int = 20) -> None:
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    resolved = [(_resolve_feature_name(name, feature_names), float(score)) for name, score in scores.items()]
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
    threshold_metrics: dict[str, list[float]], optimal_threshold: float, output_path: Path
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


__all__ = [
    "save_confusion_matrix_plot",
    "save_feature_importance_plot",
    "save_pr_curve_plot",
    "save_roc_curve_plot",
    "save_threshold_analysis_plot",
]
